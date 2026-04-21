"""Harvest a prior run_libero_eval_bitnet rollout directory into a prescreen JSON.

The prior eval iterated `initial_states[episode_idx]` over 50 indices per task
with task_order_index=0 (default task order), and recorded one MP4 per episode
named like:

    2026_04_16-09_21_21--episode=197--success=False--task=<truncated>.mp4

Episode indices are 1-based in the filenames. We parse {success} out of each
name and map episode N to (task_id, init_state_idx) via
    task_id = (N - 1) // num_trials_per_task
    init_state_idx = (N - 1) % num_trials_per_task

so the resulting prescreen JSON is schema-compatible with what
`collect_episodes.py --init_states_from` expects.

Usage:
    python -m analysis.harvest_prior_eval \\
        --rollouts_dir /home/yusei/Desktop/LIS5/BitVLA/openvla-oft/rollouts/eval_spatial_int2/2026_04_16 \\
        --suite libero_spatial \\
        --output /tmp/prescreen_libero_spatial.json
"""

import argparse
import json
import os
import re

EPISODE_RE = re.compile(r"episode=(\d+)--success=(True|False)--")


def harvest(rollouts_dir, suite, num_trials_per_task=50):
    records = []
    seen = set()
    for name in sorted(os.listdir(rollouts_dir)):
        m = EPISODE_RE.search(name)
        if not m:
            continue
        ep = int(m.group(1))
        success = m.group(2) == "True"
        if ep in seen:
            continue
        seen.add(ep)
        task_id = (ep - 1) // num_trials_per_task
        init_state_idx = (ep - 1) % num_trials_per_task
        records.append({
            "task_id": task_id,
            "init_state_idx": init_state_idx,
            "success": success,
            "episode": ep,
            "source_file": name,
        })
    return {"suite": suite, "records": records}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rollouts_dir", required=True)
    parser.add_argument("--suite", required=True, help="e.g. libero_spatial, libero_spatial_swap")
    parser.add_argument("--num_trials_per_task", type=int, default=50)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    data = harvest(args.rollouts_dir, args.suite, args.num_trials_per_task)
    recs = data["records"]
    n_success = sum(1 for r in recs if r["success"])
    n_failed = len(recs) - n_success
    print(f"Harvested {len(recs)} records from {args.rollouts_dir}")
    print(f"  success: {n_success}   failed: {n_failed}   rate: {n_failed / max(len(recs), 1):.2%}")

    tasks = sorted({r["task_id"] for r in recs})
    print(f"  task_ids: {tasks}")
    for tid in tasks:
        fails = [r for r in recs if r["task_id"] == tid and not r["success"]]
        if fails:
            print(f"    task {tid}: {len(fails)} failures at init_state_idx {[f['init_state_idx'] for f in fails]}")

    with open(args.output, "w") as f:
        json.dump(data, f, indent=2)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
