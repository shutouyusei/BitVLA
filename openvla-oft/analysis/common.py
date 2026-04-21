"""Shared helpers for BitVLA analysis scripts (attention, t-SNE).

Exposes:
- load_rollout_stack: full OFT inference stack (model, action_head, proprio_projector,
  noisy_action_projector, processor, cfg) — supports both initial-frame captures and
  live rollouts.
- capture_observation: returns an observation image under one of four modes
  (initial, success, failed, mid_rollout).
- prepare_inputs: multimodal input dict for a single image + task label.
"""

import argparse
import copy
import datetime
import os
import sys
from collections import deque
from dataclasses import dataclass

import numpy as np
import torch

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

DEVICE = torch.device("cpu")


def set_device(device_str):
    global DEVICE
    DEVICE = torch.device(device_str)
    return DEVICE


@dataclass
class RolloutStack:
    model: object
    action_head: object
    proprio_projector: object
    noisy_action_projector: object
    processor: object
    cfg: object
    resize_size: object


def _build_cfg(checkpoint_path, use_int2, task_suite_name):
    """Minimal GenerateConfig compatible with bitnet OFT inference."""
    from experiments.robot.libero.run_libero_eval_bitnet import GenerateConfig

    return GenerateConfig(
        model_family="bitnet",
        pretrained_checkpoint=checkpoint_path,
        use_l1_regression=True,
        use_diffusion=False,
        use_film=False,
        num_images_in_input=2,
        use_proprio=True,
        center_crop=True,
        num_open_loop_steps=8,
        use_int2_quantization=use_int2,
        task_suite_name=task_suite_name,
        num_steps_wait=10,
    )


def load_rollout_stack(checkpoint_path, use_int2=False, task_suite_name="libero_spatial"):
    """Load BitVLA + action head + proprio projector + processor via the OFT pipeline."""
    from experiments.robot.libero.run_libero_eval_bitnet import initialize_model
    from experiments.robot.robot_utils import get_image_resize_size

    from bitvla.constants import (
        BITNET_DEFAULT_IMAGE_TOKEN_IDX,
        BITNET_PROPRIO_PAD_IDX,
        BITNET_IGNORE_INDEX,
        BITNET_ACTION_TOKEN_BEGIN_IDX,
        BITNET_STOP_INDEX,
    )

    cfg = _build_cfg(checkpoint_path, use_int2, task_suite_name)
    model, action_head, proprio_projector, noisy_action_projector, processor = initialize_model(cfg)
    model.set_constant(
        image_token_idx=BITNET_DEFAULT_IMAGE_TOKEN_IDX,
        proprio_pad_idx=BITNET_PROPRIO_PAD_IDX,
        ignore_idx=BITNET_IGNORE_INDEX,
        action_token_begin_idx=BITNET_ACTION_TOKEN_BEGIN_IDX,
        stop_index=BITNET_STOP_INDEX,
    )
    model = model.to(DEVICE)
    model.eval()
    if action_head is not None:
        action_head = action_head.to(DEVICE).eval()
    if proprio_projector is not None:
        proprio_projector = proprio_projector.to(DEVICE).eval()
    if noisy_action_projector is not None:
        noisy_action_projector = noisy_action_projector.to(DEVICE).eval()

    resize_size = get_image_resize_size(cfg)
    return RolloutStack(
        model=model,
        action_head=action_head,
        proprio_projector=proprio_projector,
        noisy_action_projector=noisy_action_projector,
        processor=processor,
        cfg=cfg,
        resize_size=resize_size,
    )


def prepare_inputs(model, processor, image, task_label):
    """Build multimodal input dict from a single image and task description."""
    from transformers.image_utils import get_image_size, to_numpy_array
    from bitvla.constants import BITNET_DEFAULT_IMAGE_TOKEN
    from bitvla.dataset.bitvla_transform import llava_to_openai

    all_images = [image]
    pixel_values = [
        processor.image_processor.preprocess(img, return_tensors="pt")["pixel_values"][0]
        for img in all_images
    ]
    patch_size = processor.patch_size
    num_image_tokens = []
    for img in pixel_values:
        height, width = get_image_size(to_numpy_array(img))
        num_image_tokens.append((height // patch_size) * (width // patch_size))

    sources = {
        "conversations": [
            {"from": "human", "value": f"<image>\n<proprio_pad>What action should the robot take to {task_label.lower()}?"},
            {"from": "gpt", "value": ""},
        ]
    }
    sources = copy.deepcopy(llava_to_openai(sources["conversations"]))
    prompt = sources[0]["content"]
    x = [{"role": "user", "content": prompt}]
    input_str = processor.tokenizer.apply_chat_template(x, tokenize=False, add_generation_prompt=True)

    placeholder = "<TEMP_IMAGE_TOKEN>"
    input_str = input_str.replace(BITNET_DEFAULT_IMAGE_TOKEN, placeholder)
    token_index = 0
    while placeholder in input_str and token_index < len(num_image_tokens):
        input_str = input_str.replace(
            placeholder, BITNET_DEFAULT_IMAGE_TOKEN * num_image_tokens[token_index], 1
        )
        token_index += 1

    input_ids = processor.tokenizer(input_str, add_special_tokens=True).input_ids
    input_ids = torch.tensor(input_ids).unsqueeze(0)
    attention_mask = torch.ones(input_ids.shape, dtype=torch.long)
    pv = torch.stack(pixel_values, dim=0).unsqueeze(0)

    return dict(
        input_ids=input_ids.to(DEVICE, dtype=torch.long),
        attention_mask=attention_mask.to(DEVICE),
        pixel_values=pv.to(DEVICE, dtype=torch.bfloat16),
    )


def resolve_output_dirs(base_dir, subdir=None):
    """Create and return (png_dir, json_dir) under base_dir/<YYYYMMDD>/{png,json}.

    Pass --output_subdir to override the auto date folder (e.g. a fixed run name).
    """
    folder = subdir or datetime.datetime.now().strftime("%Y%m%d")
    png_dir = os.path.join(base_dir, folder, "png")
    json_dir = os.path.join(base_dir, folder, "json")
    os.makedirs(png_dir, exist_ok=True)
    os.makedirs(json_dir, exist_ok=True)
    return png_dir, json_dir


def default_layer_indices(model):
    """All BitNetAttention layers (Phase 1 design requires full layer coverage)."""
    from transformers.models.llava.modeling_bitnet import BitNetAttention

    num_layers = sum(1 for _, m in model.named_modules() if isinstance(m, BitNetAttention))
    return list(range(num_layers)), num_layers


def default_siglip_layer_indices(model):
    """All SiglipAttention layers."""
    from transformers.models.siglip.modeling_siglip import SiglipAttention

    num_layers = sum(1 for _, m in model.named_modules() if isinstance(m, SiglipAttention))
    return list(range(num_layers)), num_layers


VALID_CAPTURE_MODES = {"initial", "success", "failed", "mid_rollout"}


def parse_conditions(arg_list):
    """Parse --condition label:suite[:mode] into list of (label, suite, mode)."""
    conditions = []
    for entry in arg_list:
        parts = entry.split(":")
        if len(parts) == 2:
            label, suite = parts
            mode = "initial"
        elif len(parts) == 3:
            label, suite, mode = parts
        else:
            raise argparse.ArgumentTypeError(
                f"Invalid --condition '{entry}'. Expected 'label:suite' or 'label:suite:mode'."
            )
        if mode not in VALID_CAPTURE_MODES:
            raise argparse.ArgumentTypeError(
                f"Invalid capture mode '{mode}'. One of: {sorted(VALID_CAPTURE_MODES)}"
            )
        conditions.append((label, suite, mode))
    return conditions


def add_common_args(parser):
    parser.add_argument("--checkpoint_path", type=str, required=True)
    parser.add_argument(
        "--condition", type=str, action="append", required=True,
        help="Repeatable flag. Each entry: 'label:suite[:mode]'. "
             "mode ∈ {initial, success, failed, mid_rollout} (default: initial).",
    )
    parser.add_argument("--task_id", type=int, default=0)
    parser.add_argument("--output_dir", type=str, default="./analysis_output")
    parser.add_argument("--output_subdir", type=str, default=None,
                        help="Subdir under --output_dir (default: today's date YYYYMMDD).")
    parser.add_argument("--use_int2_quantization", action="store_true")
    parser.add_argument("--layers", type=int, nargs="*", default=None)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--rollout_max_steps", type=int, default=None,
                        help="Max env steps; None → use TASK_MAX_STEPS for the suite.")
    parser.add_argument("--rollout_seed_candidates", type=int, nargs="*",
                        default=[7, 13, 21, 42, 100])


def _get_task_and_init(suite_name, task_id):
    from libero.libero import benchmark
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[suite_name]()
    task = task_suite.get_task(task_id)
    initial_states = task_suite.get_task_init_states(task_id)
    if initial_states is None or len(initial_states) == 0:
        raise RuntimeError(f"No initial states for task {task_id} in suite '{suite_name}'.")
    return task, initial_states[0], task.language, task.name


def _make_seg_env(task, resolution=256):
    """Construct a LIBERO env with instance segmentation enabled."""
    from libero.libero import get_libero_path
    from libero.libero.envs import SegmentationRenderEnv

    bddl = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
    env = SegmentationRenderEnv(
        bddl_file_name=bddl,
        camera_heights=resolution,
        camera_widths=resolution,
        camera_segmentations="instance",
    )
    env.seed(0)
    return env


_ROBOT_NAME_KEYWORDS = ("panda", "gripper", "mount", "rethink")


def _is_robot_name(name: str) -> bool:
    lower = name.lower()
    return any(kw in lower for kw in _ROBOT_NAME_KEYWORDS)


def _extract_bboxes(env, obs, flip=True):
    """Return [{name, bbox, is_target, is_robot}] for all scene objects.

    is_robot flags Panda/Gripper/Mount/Rethink entries so downstream metrics
    can exclude the robot (per Phase 1 design: '計測対象物体: 全物体（ロボット除く）').
    """
    seg = obs.get("agentview_segmentation_instance")
    if seg is None:
        return []
    seg = np.squeeze(np.asarray(seg))
    if flip:
        seg = seg[::-1, ::-1]
    target_names = set(getattr(env, "obj_of_interest", None) or [])
    bboxes = []
    for inst_id, name in env.segmentation_id_mapping.items():
        mask = (seg == inst_id + 1)
        if not mask.any():
            continue
        rows = np.any(mask, axis=1)
        cols = np.any(mask, axis=0)
        ymin, ymax = np.where(rows)[0][[0, -1]]
        xmin, xmax = np.where(cols)[0][[0, -1]]
        bboxes.append({
            "name": name,
            "bbox": (int(ymin), int(xmin), int(ymax), int(xmax)),
            "is_target": name in target_names,
            "is_robot": _is_robot_name(name),
        })
    return bboxes


def _reset_and_capture_initial(suite_name, task_id, seed):
    from experiments.robot.libero.libero_utils import get_libero_image
    task, init_state, task_description, task_name = _get_task_and_init(suite_name, task_id)
    env = _make_seg_env(task, resolution=256)
    try:
        env.seed(seed)
        env.reset()
        env.set_init_state(init_state)
        obs = env.step(np.zeros(7))[0]
        image = get_libero_image(obs)
        bboxes = _extract_bboxes(env, obs, flip=True)
    finally:
        env.close()
    return image, task_description, task_name, bboxes


PHASE1_CAPTURE_STEPS = {"t=0": 0, "t=max//3": None, "step=100": 100, "t=max-1": None}


def _resolve_capture_steps(max_steps, override=None):
    """Return concrete {frame_name: step_idx} for a given rollout length."""
    steps = dict(override or PHASE1_CAPTURE_STEPS)
    resolved = {}
    for name, raw in steps.items():
        if name == "t=max//3":
            resolved[name] = max_steps // 3
        elif name == "t=max-1":
            resolved[name] = max_steps - 1
        else:
            resolved[name] = raw
    return resolved


def _run_rollout(stack: RolloutStack, suite_name, task_id, seed, max_steps, capture_steps):
    """Run a rollout, collecting image+bbox at every requested frame.

    capture_steps: dict {frame_name: step_idx}. Frames that the rollout never
    reaches (rollout terminates early) are filled from the final captured
    observation and tagged with fallback=True.
    """
    from experiments.robot.libero.libero_utils import (
        get_libero_image,
        get_libero_dummy_action,
    )
    from experiments.robot.libero.run_libero_eval_bitnet import (
        prepare_observation,
        process_action,
        TASK_MAX_STEPS,
    )
    from experiments.robot.robot_utils import get_action

    task, init_state, task_description, task_name = _get_task_and_init(suite_name, task_id)
    cfg = stack.cfg
    cfg.task_suite_name = suite_name
    if max_steps is None:
        max_steps = TASK_MAX_STEPS.get(suite_name, 220)

    env = _make_seg_env(task, resolution=256)
    frames = {}
    last_frame = None
    action_queue = deque(maxlen=cfg.num_open_loop_steps)
    success = False
    try:
        env.seed(seed)
        env.reset()
        obs = env.set_init_state(init_state)
        t = 0
        while t < max_steps + cfg.num_steps_wait:
            if t < cfg.num_steps_wait:
                obs, _, _, _ = env.step(get_libero_dummy_action(cfg.model_family))
                t += 1
                continue

            step_idx = t - cfg.num_steps_wait
            last_frame = {
                "image": get_libero_image(obs),
                "bboxes": _extract_bboxes(env, obs, flip=True),
                "step_idx": step_idx,
            }
            for fname, tgt in capture_steps.items():
                if tgt is not None and step_idx == tgt and fname not in frames:
                    frames[fname] = dict(last_frame)

            observation, _ = prepare_observation(obs, stack.resize_size)
            if len(action_queue) == 0:
                actions = get_action(
                    cfg, stack.model, observation, task_description,
                    processor=stack.processor,
                    action_head=stack.action_head,
                    proprio_projector=stack.proprio_projector,
                    noisy_action_projector=stack.noisy_action_projector,
                    use_film=cfg.use_film,
                )
                action_queue.extend(actions)
            action = process_action(action_queue.popleft(), cfg.model_family)
            obs, _, done, _ = env.step(action.tolist())
            if done:
                success = True
                break
            t += 1
    finally:
        env.close()

    if last_frame is None:
        raise RuntimeError(f"Rollout for '{suite_name}' task {task_id} produced no observation.")

    for fname in capture_steps:
        if fname not in frames:
            frames[fname] = {**last_frame, "fallback": True}

    return {
        "success": success,
        "frames": frames,
        "final_step_idx": last_frame["step_idx"],
        "task_description": task_description,
        "task_name": task_name,
    }


def capture_frames(
    suite_name, task_id, mode="initial", *,
    stack: RolloutStack = None,
    seed=7,
    rollout_max_steps=None,
    rollout_seed_candidates=(7, 13, 21, 42, 100),
    capture_steps=None,
):
    """Return {task_description, task_name, meta, frames:{frame_name: {image, bboxes, step_idx}}}.

    Modes:
      initial     : returns a single frame at t=0 without running the policy.
      success     : runs rollouts until one succeeds; returns all 4 frames.
      failed      : runs rollouts until one fails; returns all 4 frames.
      mid_rollout : runs one rollout with `seed`; returns all 4 frames regardless.
    """
    if mode == "initial":
        image, task_desc, task_name, bboxes = _reset_and_capture_initial(suite_name, task_id, seed)
        return {
            "task_description": task_desc,
            "task_name": task_name,
            "meta": {"mode": mode, "seed": seed},
            "frames": {"t=0": {"image": np.array(image), "bboxes": bboxes, "step_idx": 0}},
        }

    if stack is None:
        raise ValueError(f"mode='{mode}' requires a RolloutStack (pass stack=...).")

    from experiments.robot.libero.run_libero_eval_bitnet import TASK_MAX_STEPS
    max_steps = rollout_max_steps or TASK_MAX_STEPS.get(suite_name, 220)
    steps = _resolve_capture_steps(max_steps, capture_steps)

    def _package(result, meta):
        packaged = {}
        for fname, frame in result["frames"].items():
            packaged[fname] = {
                "image": np.array(frame["image"]),
                "bboxes": frame["bboxes"],
                "step_idx": frame["step_idx"],
                "fallback": bool(frame.get("fallback", False)),
            }
        return {
            "task_description": result["task_description"],
            "task_name": result["task_name"],
            "meta": meta,
            "frames": packaged,
        }

    if mode == "mid_rollout":
        result = _run_rollout(stack, suite_name, task_id, seed, max_steps, steps)
        return _package(result, {"mode": mode, "seed": seed, "success": result["success"],
                                  "final_step_idx": result["final_step_idx"]})

    target_success = (mode == "success")
    for candidate_seed in rollout_seed_candidates:
        print(f"  trying seed={candidate_seed} for mode={mode}...")
        result = _run_rollout(stack, suite_name, task_id, candidate_seed, max_steps, steps)
        if result["success"] == target_success:
            return _package(result, {"mode": mode, "seed": candidate_seed,
                                      "success": result["success"],
                                      "final_step_idx": result["final_step_idx"]})
    raise RuntimeError(
        f"No rollout seed in {list(rollout_seed_candidates)} produced outcome '{mode}' "
        f"for suite '{suite_name}' task {task_id}."
    )
