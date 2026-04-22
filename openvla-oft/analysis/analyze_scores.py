"""Aggregate episode JSONs and run pairwise Mann-Whitney U with Bonferroni.

Primary metric (per Phase 1 design):
    target_score(layer) = Σ ratio[i] for i with is_target=True and is_robot=False
That scalar is computed per episode per frame per layer for both the LLM side
and the SigLIP side. Across episodes within a condition we summarize mean/std/N,
and between conditions we run Mann-Whitney U (asymptotic) with Bonferroni α
and effect size r = |z| / √(n1 + n2).

Usage:
    python -m analysis.analyze_scores \\
        --episodes_dir analysis_output/<YYYYMMDD>/json/episodes \\
        --output_dir   analysis_output/<YYYYMMDD>/json/summary
"""

import argparse
import glob
import itertools
import json
import math
import os
from collections import defaultdict

import numpy as np
from scipy import stats


def target_score_from_ratios(ratios_per_layer):
    """Per-layer sum of is_target (non-robot) ratios. NaN if no non-robot target objects."""
    out = {}
    for layer_idx, per_obj in ratios_per_layer.items():
        total = 0.0
        n_targets = 0
        for _name, info in per_obj.items():
            if info.get("is_target") and not info.get("is_robot"):
                r = info["ratio"]
                if r is None or (isinstance(r, float) and math.isnan(r)):
                    continue
                total += r
                n_targets += 1
        out[int(layer_idx)] = total if n_targets > 0 else float("nan")
    return out


def load_episodes(episodes_dir):
    by_label = defaultdict(list)
    for path in sorted(glob.glob(os.path.join(episodes_dir, "*.json"))):
        try:
            ep = json.load(open(path))
        except (OSError, json.JSONDecodeError):
            continue
        by_label[ep["label"]].append(ep)
    return by_label


def aggregate(episodes):
    """Return {frame_name: {side: {layer: [target_scores across episodes]}}}."""
    per_frame = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for ep in episodes:
        for frame_name, fr in ep["frames"].items():
            if "per_layer_ratios_llm" in fr:
                for layer, score in target_score_from_ratios(fr["per_layer_ratios_llm"]).items():
                    per_frame[frame_name]["llm"][layer].append(score)
            if "per_layer_ratios_siglip" in fr:
                for layer, score in target_score_from_ratios(fr["per_layer_ratios_siglip"]).items():
                    per_frame[frame_name]["siglip"][layer].append(score)
    return per_frame


def summarize_values(values_by_layer):
    out = {}
    for layer, vals in values_by_layer.items():
        arr = np.asarray([v for v in vals if not (isinstance(v, float) and math.isnan(v))])
        out[int(layer)] = {
            "mean": float(arr.mean()) if arr.size else None,
            "std": float(arr.std(ddof=1)) if arr.size > 1 else None,
            "n": int(arr.size),
            "values": arr.tolist(),
        }
    return out


def mann_whitney_with_effect(x, y):
    """Two-sided Mann-Whitney U, asymptotic (tie-corrected).

    Returns {u, p, z, r, n1, n2}. z is derived from scipy's tie-corrected p-value
    (z = Φ⁻¹(1 − p/2) with sign from U − n1·n2/2) so r = |z| / √(n1 + n2) remains
    consistent with p under rank ties. The untied closed-form overstates |z| and
    therefore r when many values tie (e.g. zero-attention bboxes).
    """
    x = np.asarray([v for v in x if not (isinstance(v, float) and math.isnan(v))])
    y = np.asarray([v for v in y if not (isinstance(v, float) and math.isnan(v))])
    n1, n2 = int(x.size), int(y.size)
    if n1 < 2 or n2 < 2:
        return {"u": None, "p": None, "z": None, "r": None, "n1": n1, "n2": n2}
    res = stats.mannwhitneyu(x, y, alternative="two-sided", method="asymptotic")
    u = float(res.statistic)
    p = float(res.pvalue)
    mean_u = n1 * n2 / 2.0
    sign = 1.0 if u > mean_u else (-1.0 if u < mean_u else 0.0)
    if p <= 0.0:
        z = sign * float("inf")
    elif p >= 1.0:
        z = 0.0
    else:
        z = sign * float(stats.norm.isf(p / 2.0))
    r = abs(z) / math.sqrt(n1 + n2) if math.isfinite(z) else float("inf")
    return {"u": u, "p": p, "z": float(z), "r": float(r), "n1": n1, "n2": n2}


def run_tests(aggregated_raw):
    labels = sorted(aggregated_raw.keys())
    pairs = list(itertools.combinations(labels, 2))
    alpha = 0.05 / max(len(pairs), 1)

    frames = set()
    for cond in aggregated_raw.values():
        frames.update(cond.keys())

    out = {
        "labels": labels,
        "pair_count": len(pairs),
        "alpha_bonferroni": alpha,
        "tests": {},
    }
    for frame in sorted(frames):
        out["tests"][frame] = {}
        for side in ("llm", "siglip"):
            layers = set()
            for label in labels:
                layers.update(aggregated_raw[label].get(frame, {}).get(side, {}).keys())
            if not layers:
                continue
            out["tests"][frame][side] = {}
            for layer in sorted(layers):
                per_pair = {}
                for a, b in pairs:
                    xs = aggregated_raw[a].get(frame, {}).get(side, {}).get(layer, [])
                    ys = aggregated_raw[b].get(frame, {}).get(side, {}).get(layer, [])
                    res = mann_whitney_with_effect(xs, ys)
                    if res["p"] is not None:
                        res["sig_bonferroni"] = bool(res["p"] < alpha)
                    per_pair[f"{a}_vs_{b}"] = res
                out["tests"][frame][side][int(layer)] = per_pair
    return out


def print_summary(tests):
    alpha = tests["alpha_bonferroni"]
    print(f"\nBonferroni α = 0.05 / {tests['pair_count']} = {alpha:.5f}\n")
    for frame, side_map in tests["tests"].items():
        for side, layer_map in side_map.items():
            sig = [(l, pair, r) for l, pair_map in layer_map.items()
                   for pair, r in pair_map.items()
                   if r.get("sig_bonferroni")]
            if not sig:
                print(f"[{frame} / {side}]  no significant layers after Bonferroni")
                continue
            print(f"[{frame} / {side}]  {len(sig)} significant (layer, pair):")
            for layer, pair, r in sig[:20]:
                print(f"    layer {layer:>2} {pair:30s} p={r['p']:.2e} r={r['r']:.2f} (n={r['n1']}, {r['n2']})")
            if len(sig) > 20:
                print(f"    ... ({len(sig) - 20} more)")


def main():
    parser = argparse.ArgumentParser(description="Phase 1 statistical aggregator")
    parser.add_argument("--episodes_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    by_label = load_episodes(args.episodes_dir)
    total = sum(len(v) for v in by_label.values())
    print(f"Loaded {total} episodes across {len(by_label)} conditions:")
    for label, eps in by_label.items():
        print(f"  {label}: n={len(eps)}")
    if total == 0:
        print("No episodes found. Nothing to do.")
        return

    aggregated_raw = {}
    for label, episodes in by_label.items():
        agg = aggregate(episodes)
        summarized = {
            frame: {side: summarize_values(layer_map) for side, layer_map in side_map.items()}
            for frame, side_map in agg.items()
        }
        payload = {
            "label": label,
            "n_episodes": len(episodes),
            "task_ids": sorted({ep["task_id"] for ep in episodes}),
            "init_state_indices": sorted({ep.get("init_state_idx") for ep in episodes
                                          if "init_state_idx" in ep}),
            "frames": summarized,
        }
        out_path = os.path.join(args.output_dir, f"aggregated_{label}.json")
        with open(out_path, "w") as f:
            json.dump(payload, f, indent=2)
        print(f"Wrote {out_path}")
        aggregated_raw[label] = agg

    tests = run_tests(aggregated_raw)
    tests_path = os.path.join(args.output_dir, "stat_tests.json")
    with open(tests_path, "w") as f:
        json.dump(tests, f, indent=2)
    print(f"Wrote {tests_path}")

    print_summary(tests)


if __name__ == "__main__":
    main()
