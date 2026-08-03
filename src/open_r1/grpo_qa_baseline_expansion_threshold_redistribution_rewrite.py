import numpy as np
import os
import re
from datetime import datetime
from dataclasses import dataclass, field
from typing import Optional

from datasets import Dataset, DatasetDict
from transformers import Qwen2VLForConditionalGeneration, set_seed
from transformers import Qwen2_5_VLForConditionalGeneration
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import get_peft_config

from src.open_r1.trainer import Qwen2VLGRPOTrainer_Video_QA as Qwen2VLGRPOTrainer
from src.open_r1.uncertainty_consensus import (
    rewards_from_consensus_distribution,
    uncertainty_aware_consensus_distribution,
    uncertainty_consensus_from_entropies,
)
from trl import GRPOConfig, ModelConfig, ScriptArguments, TrlParser
from src.open_r1.my_qwen_utils_opencv import process_vision_info

import torch
import random
import ast
import csv
import gc

DEFAULT_VIDEO_MAX_PIXELS = 768 * 28 * 28
DEFAULT_VIDEO_MIN_PIXELS = 16 * 28 * 28
DEFAULT_VIDEO_FPS = 1.0
DEFAULT_VIDEO_MAX_FRAMES = 60


# =========================
# Arguments
# =========================
@dataclass
class GRPOScriptArguments(ScriptArguments):
    reward_funcs: list[str] = field(
        default_factory=lambda: ["answer"]
    )
    # Keep the inference script's 1fps/full-video sampling style, but cap
    # frames and per-frame pixels so GRPO rollouts fit in memory.
    max_pixels: Optional[int] = DEFAULT_VIDEO_MAX_PIXELS
    min_pixels: Optional[int] = DEFAULT_VIDEO_MIN_PIXELS

    max_frames: Optional[int] = DEFAULT_VIDEO_MAX_FRAMES
    video_fps: Optional[float] = DEFAULT_VIDEO_FPS

    train_data_path: str = ""

    train_video_folder: str = ""
    use_think_prompt: bool = False

from collections import Counter

_STANDALONE_OPT_RE = re.compile(r'(?<![A-Za-z0-9])([A-D])(?![A-Za-z0-9])', re.I)


def extract_choice_answer(text: str) -> Optional[str]:
    if text is None:
        return None

    m = re.search(r'<answer>\s*([A-D])\s*</answer>', text, re.I)
    if m:
        return m.group(1).upper()

    m = re.search(r'<\s*([A-D])\s*/?\s*>', text, re.I)
    if m:
        return m.group(1).upper()

    parts = re.split(r'</think>', text, flags=re.I)
    tail = parts[-1] if len(parts) > 1 else ''
    opts_in_tail = _STANDALONE_OPT_RE.findall(tail)
    if opts_in_tail:
        return opts_in_tail[-1].upper()

    opts_all = _STANDALONE_OPT_RE.findall(text)
    if opts_all:
        return opts_all[-1].upper()

    return None


def answer_reward(completions, solution=None, **kwargs):
    """
    Uncertainty-aware consensus reward for multiple-choice QA.

    Core design:
    1. Each sample's entropy H_i is computed from its original raw completion
       token sequence by the trainer, then passed in as mean_token_entropies.
    2. Consensus grouping is performed using the parsed option label A/B/C/D,
       not the raw completion string.
    3. Invalid completions that cannot be parsed into A/B/C/D receive reward 0
       and do not participate in the consensus distribution.

    Therefore, the following raw outputs are grouped as the same answer B:
        "B"
        " B"
        "The answer is B"
        "<answer>B</answer>"
    while their H_i / w_i may still differ because they have different
    original token sequences.
    """

    # ------------------------------------------------------------------
    # 1. Preserve raw completions for logging/debugging.
    # ------------------------------------------------------------------
    raw_completions = [
        "" if completion is None else str(completion)
        for completion in completions
    ]

    if not raw_completions:
        return []

    # ------------------------------------------------------------------
    # 2. Parse each raw completion into A/B/C/D.
    #    `extract_choice_answer` is already defined in this training script.
    # ------------------------------------------------------------------
    parsed_answers = [
        extract_choice_answer(completion)
        for completion in raw_completions
    ]

    # Invalid outputs receive reward 0 and do not affect consensus.
    rewards = [0.0] * len(raw_completions)

    valid_indices = [
        index
        for index, answer in enumerate(parsed_answers)
        if answer is not None
    ]

    # ------------------------------------------------------------------
    # 3. Read hyperparameters.
    # ------------------------------------------------------------------
    lambda_ = float(
        kwargs.get(
            "uncertainty_lambda",
            os.environ.get("GRPO_CONSENSUS_LAMBDA", 1.0),
        )
    )
    tau = float(
        kwargs.get(
            "consensus_tau",
            os.environ.get("GRPO_CONSENSUS_TAU", 0.2),
        )
    )
    delta = float(
        kwargs.get(
            "consensus_delta",
            os.environ.get("GRPO_CONSENSUS_DELTA", 0.05),
        )
    )
    eps = float(
        kwargs.get(
            "consensus_eps",
            os.environ.get("GRPO_CONSENSUS_EPS", 1e-12),
        )
    )
    empty_high_policy = kwargs.get(
        "empty_high_policy",
        os.environ.get(
            "GRPO_CONSENSUS_EMPTY_HIGH_POLICY",
            "return_p0",
        ),
    )

    mean_token_entropies = kwargs.get("mean_token_entropies")
    token_probs = kwargs.get("token_probs")
    consensus_debug_records = kwargs.get("consensus_debug_records")
    return_debug = consensus_debug_records is not None

    # ------------------------------------------------------------------
    # 4. Validate that entropy/probability inputs align with completions.
    # ------------------------------------------------------------------
    if mean_token_entropies is None and token_probs is None:
        raise ValueError(
            "answer_reward requires mean_token_entropies or token_probs."
        )

    if mean_token_entropies is not None:
        if len(mean_token_entropies) != len(raw_completions):
            raise ValueError(
                "mean_token_entropies must have the same length as completions."
            )

    if token_probs is not None:
        if len(token_probs) != len(raw_completions):
            raise ValueError(
                "token_probs must have the same length as completions."
            )

    # ------------------------------------------------------------------
    # 5. If no valid A/B/C/D answer exists, return all-zero rewards.
    # ------------------------------------------------------------------
    if not valid_indices:
        if consensus_debug_records is not None:
            debug_H = (
                [float(h) for h in mean_token_entropies]
                if mean_token_entropies is not None
                else [None] * len(raw_completions)
            )

            consensus_debug_records.append(
                {
                    "lambda": lambda_,
                    "tau": tau,
                    "delta": delta,
                    "eps": eps,
                    "empty_high_policy": empty_high_policy,
                    "parsed_answers": parsed_answers,
                    "valid_sample_indices": [],
                    "H": debug_H,
                    "w": [None] * len(raw_completions),
                    "R": {},
                    "p0": {},
                    "p": {},
                    "A_high": [],
                    "A_low": [],
                    "samples": [
                        {
                            "sample_index": index,
                            "answer": None,
                            "completion": raw_completions[index],
                            "valid_for_consensus": False,
                            "H": debug_H[index],
                            "w": None,
                            "R": None,
                            "p0": None,
                            "p": None,
                            "reward": 0.0,
                        }
                        for index in range(len(raw_completions))
                    ],
                }
            )

        return rewards

    # ------------------------------------------------------------------
    # 6. Use parsed A/B/C/D for grouping, but retain each original sample's
    #    corresponding entropy / probability matrix.
    # ------------------------------------------------------------------
    valid_answers = [
        parsed_answers[index]
        for index in valid_indices
    ]

    if mean_token_entropies is not None:
        valid_mean_token_entropies = [
            mean_token_entropies[index]
            for index in valid_indices
        ]

        consensus_result = uncertainty_consensus_from_entropies(
            answers=valid_answers,
            mean_entropies=valid_mean_token_entropies,
            lambda_=lambda_,
            tau=tau,
            delta=delta,
            eps=eps,
            empty_high_policy=empty_high_policy,
            return_debug=return_debug,
        )

    else:
        valid_token_probs = [
            token_probs[index]
            for index in valid_indices
        ]

        consensus_result = uncertainty_aware_consensus_distribution(
            answers=valid_answers,
            token_probs=valid_token_probs,
            lambda_=lambda_,
            tau=tau,
            delta=delta,
            eps=eps,
            empty_high_policy=empty_high_policy,
            return_debug=return_debug,
        )

    if return_debug:
        final_p, debug = consensus_result
    else:
        final_p = consensus_result
        debug = None

    # ------------------------------------------------------------------
    # 7. Map valid rewards back to their original completion positions.
    # ------------------------------------------------------------------
    valid_rewards = rewards_from_consensus_distribution(
        valid_answers,
        final_p,
    )

    for original_index, reward in zip(valid_indices, valid_rewards):
        rewards[original_index] = float(reward)

    # ------------------------------------------------------------------
    # 8. Save aligned debug information.
    # ------------------------------------------------------------------
    if consensus_debug_records is not None:
        valid_position_by_original_index = {
            original_index: valid_position
            for valid_position, original_index in enumerate(valid_indices)
        }

        debug_by_answer = {
            answer: {
                "R": float(debug.consensus_scores[answer]),
                "p0": float(debug.p0[answer]),
                "p": float(final_p[answer]),
            }
            for answer in debug.p0
        }

        aligned_H = [None] * len(raw_completions)
        aligned_w = [None] * len(raw_completions)

        for valid_position, original_index in enumerate(valid_indices):
            aligned_H[original_index] = float(
                debug.mean_entropies[valid_position]
            )
            aligned_w[original_index] = float(
                debug.weights[valid_position]
            )

        samples = []

        for sample_index, raw_completion in enumerate(raw_completions):
            parsed_answer = parsed_answers[sample_index]
            is_valid = sample_index in valid_position_by_original_index

            if is_valid:
                answer_debug = debug_by_answer[parsed_answer]

                samples.append(
                    {
                        "sample_index": sample_index,
                        "answer": parsed_answer,
                        "completion": raw_completion,
                        "valid_for_consensus": True,
                        "H": aligned_H[sample_index],
                        "w": aligned_w[sample_index],
                        "R": answer_debug["R"],
                        "p0": answer_debug["p0"],
                        "p": answer_debug["p"],
                        "reward": float(rewards[sample_index]),
                    }
                )
            else:
                samples.append(
                    {
                        "sample_index": sample_index,
                        "answer": None,
                        "completion": raw_completion,
                        "valid_for_consensus": False,
                        "H": (
                            float(mean_token_entropies[sample_index])
                            if mean_token_entropies is not None
                            else None
                        ),
                        "w": None,
                        "R": None,
                        "p0": None,
                        "p": None,
                        "reward": 0.0,
                    }
                )

        consensus_debug_records.append(
            {
                "lambda": lambda_,
                "tau": tau,
                "delta": delta,
                "eps": eps,
                "empty_high_policy": empty_high_policy,
                "parsed_answers": parsed_answers,
                "valid_sample_indices": valid_indices,
                "H": aligned_H,
                "w": aligned_w,
                "R": debug.consensus_scores,
                "p0": debug.p0,
                "p": final_p,
                "A_high": sorted(debug.high_answers),
                "A_low": sorted(debug.low_answers),
                "samples": samples,
            }
        )

    return rewards

reward_funcs_registry = {
    "answer": answer_reward
}


def load_csv_dataset(train_data_path, train_video_folder, seed):

    def create_dataset(file_path, video_folder):
        examples = []
        with open(file_path, mode='r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                options = [
                    "A. " + row['Option 1'],
                    "B. " + row['Option 2'],
                    "C. " + row['Option 3'],
                    "D. " + row['Option 4']
                ]

                video_name = row['Video Name']
                video_path = os.path.join(video_folder, video_name)

                examples.append({
                    "problem": {
                        "question": row['Question'],
                        "rewritten_question": row.get('Rewritten Question', '').strip(),
                        "options": options
                    },
                    "solution": {
                        "answer": row['Correct Option']
                    },
                    "video_path": video_path,
                })
        set_seed(seed)
        random.shuffle(examples)
        return Dataset.from_list(examples)

    return DatasetDict({
        "test": create_dataset(train_data_path, train_video_folder),
    })

# =========================
# Lazy Video Loader
# =========================
def load_video_lazy(
    video_path,
    max_frames=DEFAULT_VIDEO_MAX_FRAMES,
    fps=DEFAULT_VIDEO_FPS,
    max_pixels=DEFAULT_VIDEO_MAX_PIXELS,
    min_pixels=DEFAULT_VIDEO_MIN_PIXELS,
):
    try:
        max_frames = max(2, int(max_frames))
        # my_qwen_utils allocates a per-frame budget from total_pixels / nframes.
        # This keeps long videos from silently becoming much heavier than eval.
        total_pixels = int(max_pixels * max_frames / 2)
        messages = [{
            "role": "user",
            "content": [{
                "type": "video",
                "video": video_path,
                "fps": float(fps),
                "max_frames": max_frames,
                "max_pixels": int(max_pixels),
                "min_pixels": int(min_pixels),
                "total_pixels": total_pixels,
            }]
        }]

        image_inputs, video_inputs, video_kwargs = process_vision_info(
            [messages],
            return_video_kwargs=True
        )

        return video_inputs, video_kwargs

    except Exception as e:
        print(f"[WARNING] video load failed: {video_path}, error: {e}")
        return None, None


# =========================
# Main
# =========================
def main(script_args, training_args, model_args):

    reward_funcs = [reward_funcs_registry[f] for f in script_args.reward_funcs]
    data_seed = training_args.data_seed if training_args.data_seed is not None else training_args.seed
    dataset = load_csv_dataset(
        script_args.train_data_path,
        script_args.train_video_folder,
        data_seed
    )

    if not training_args.use_vllm:
        trainer_cls = Qwen2VLGRPOTrainer
    else:
        raise NotImplementedError

    # print("Using trainer:", trainer_cls)

    # Initialize the trainer.
    trainer = trainer_cls(
        model=model_args.model_name_or_path,
        reward_funcs=reward_funcs,
        args=training_args,
        train_dataset=dataset["test"],
        eval_dataset=None,
        peft_config=None,
        attn_implementation=model_args.attn_implementation,
        max_pixels=script_args.max_pixels,
        min_pixels=script_args.min_pixels,
        use_think_prompt=script_args.use_think_prompt,
        dataset_name=script_args.dataset_name,
    )

    original_collate = trainer.data_collator

    def safe_collate(features):
        for f in features:
            video_path = f["video_path"]

            video_inputs, video_kwargs = load_video_lazy(
                video_path,
                max_frames=script_args.max_frames,
                fps=script_args.video_fps,
                max_pixels=script_args.max_pixels,
                min_pixels=script_args.min_pixels,
            )

            f["video_inputs"] = video_inputs
            f["video_kwargs"] = video_kwargs

        batch = original_collate(features)

        gc.collect()

        return batch

    trainer.data_collator = safe_collate

    # =========================
    # Train
    # =========================
    trainer.train()

    trainer.save_model(training_args.output_dir)


# =========================
# Entry
# =========================
if __name__ == "__main__":
    parser = TrlParser((GRPOScriptArguments, GRPOConfig, ModelConfig))
    script_args, training_args, model_args = parser.parse_args_and_config()

    main(script_args, training_args, model_args)
