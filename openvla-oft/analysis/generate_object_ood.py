"""Generate libero_spatial_object_ood BDDL + init files.

LIBERO-PRO ships the benchmark name `libero_spatial_object_ood` in
`libero_suite_task_map.py` but does not include pre-built BDDL / init_state
files for it. This script materializes them by applying single-object
substitutions (one per variant) to the base libero_spatial BDDLs, then
invokes LIBERO-PRO's `generate_init_states.py` to produce the init files.

Each _ood task name has the form `<base_task>(<substitution>)` where
`<substitution>` is the NEW asset name (e.g. `yellow_plate`). The class it
replaces is looked up in `SUBSTITUTION_MAP` below.

Usage:
    python -m analysis.generate_object_ood \\
        --bddl_base /home/yusei/Desktop/LIS5/LIBERO-PRO/libero/libero/bddl_files \\
        --init_base /home/yusei/Desktop/LIS5/LIBERO-PRO/libero/libero/init_files \\
        --suite libero_spatial
"""

import argparse
import os
import re
import subprocess
import sys
import tempfile

import yaml

# asset substitution → class name it replaces (from LIBERO-PRO asset inventory)
SUBSTITUTION_MAP = {
    # bowls — all replace the default akita_black_bowl class
    "black_bowl": "akita_black_bowl",
    "red_bowl": "akita_black_bowl",
    "white_bowl": "akita_black_bowl",
    "yellow_bowl": "akita_black_bowl",
    "bigger_akita_black_bowl": "akita_black_bowl",
    "red_akita_black_bowl": "akita_black_bowl",
    # plate
    "yellow_plate": "plate",
    # ramekin
    "red_ramekin": "glazed_rim_porcelain_ramekin",
    # cookies
    "yellow_cookies": "cookies",
    # cabinets
    "yellow_cabinet": "wooden_cabinet",
    "white_cabinet": "wooden_cabinet",
    # stove
    "yellow_stove": "flat_stove",
}

LANG_BLOCK_RE = re.compile(r"\(:language\b.*?\)", re.S)


def apply_single_substitution(content: str, old_name: str, new_name: str) -> str:
    """Replace old_name → new_name everywhere except inside the :language block.

    Matches ObjectReplacePerturbator's language-preserving behavior so we don't
    rewrite the natural-language instruction when only the asset changed.
    """
    m = LANG_BLOCK_RE.search(content)
    if m:
        prefix = content[:m.start()].replace(old_name, new_name)
        suffix = content[m.end():].replace(old_name, new_name)
        return prefix + m.group(0) + suffix
    return content.replace(old_name, new_name)


def parse_task_list(suite_name):
    """Return [(base_task, substitution)] deduplicated, for <suite>_object_ood."""
    from libero.libero.benchmark.libero_suite_task_map import libero_task_map

    key = f"{suite_name}_object_ood"
    raw = libero_task_map.get(key, [])
    pairs = []
    seen = set()
    for entry in raw:
        e = entry.strip().rstrip(",").strip()
        if e.endswith(".bddl"):
            e = e[: -len(".bddl")]
        m = re.match(r"^(.+?)\((\w+)\)$", e)
        if not m:
            print(f"  warn: couldn't parse task entry {entry!r}", file=sys.stderr)
            continue
        pair = (m.group(1), m.group(2))
        if pair in seen:
            continue
        seen.add(pair)
        pairs.append(pair)
    return pairs


def build_bddls(suite_name, bddl_base, out_bddl_dir):
    pairs = parse_task_list(suite_name)
    print(f"Parsed {len(pairs)} unique (task, substitution) pairs for {suite_name}_object_ood")

    os.makedirs(out_bddl_dir, exist_ok=True)
    base_dir = os.path.join(bddl_base, suite_name)
    built = 0
    for base_task, sub in pairs:
        if sub not in SUBSTITUTION_MAP:
            print(f"  skip: no SUBSTITUTION_MAP entry for '{sub}' (task {base_task})")
            continue
        orig_class = SUBSTITUTION_MAP[sub]
        base_bddl = os.path.join(base_dir, f"{base_task}.bddl")
        if not os.path.exists(base_bddl):
            print(f"  skip: missing base BDDL {base_bddl}")
            continue
        with open(base_bddl) as f:
            content = f.read()
        new_content = apply_single_substitution(content, orig_class, sub)
        out_path = os.path.join(out_bddl_dir, f"{base_task}({sub}).bddl")
        with open(out_path, "w") as f:
            f.write(new_content)
        built += 1
    print(f"Wrote {built} BDDL files to {out_bddl_dir}")
    return built


def generate_init_states(out_bddl_dir, out_init_dir, script_path):
    os.makedirs(out_init_dir, exist_ok=True)
    cmd = [
        sys.executable, script_path,
        "--bddl_base_dir", out_bddl_dir,
        "--output_dir", out_init_dir,
    ]
    print(f"Running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bddl_base", type=str, required=True,
                        help="LIBERO-PRO bddl_files/ root")
    parser.add_argument("--init_base", type=str, required=True,
                        help="LIBERO-PRO init_files/ root")
    parser.add_argument("--suite", type=str, default="libero_spatial")
    parser.add_argument("--script_path", type=str,
                        default="/home/yusei/Desktop/LIS5/LIBERO-PRO/notebooks/generate_init_states.py")
    parser.add_argument("--skip_init", action="store_true",
                        help="Write BDDLs only; skip init_state generation.")
    args = parser.parse_args()

    out_bddl_dir = os.path.join(args.bddl_base, f"{args.suite}_object_ood")
    out_init_dir = os.path.join(args.init_base, f"{args.suite}_object_ood")

    n_built = build_bddls(args.suite, args.bddl_base, out_bddl_dir)
    if n_built == 0:
        print("No BDDLs generated — aborting init-state step.")
        return

    if args.skip_init:
        print("Skipping init_state generation (per --skip_init).")
        return

    generate_init_states(out_bddl_dir, out_init_dir, args.script_path)
    print(f"\nDone. BDDLs: {out_bddl_dir}\n      Init:  {out_init_dir}")


if __name__ == "__main__":
    main()
