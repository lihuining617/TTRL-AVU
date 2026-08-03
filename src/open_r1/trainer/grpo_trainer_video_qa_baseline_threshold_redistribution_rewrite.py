import fcntl
import csv
import gc
import json
import os
import re
import textwrap
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable, Optional, Union
from torch.nn.utils.rnn import pad_sequence
import torch
import torch.utils.data
import transformers
from datasets import Dataset, IterableDataset
from packaging import version
from transformers import (
    AriaForConditionalGeneration,
    AriaProcessor,
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoProcessor,
    AutoTokenizer,
    GenerationConfig,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    Qwen2VLForConditionalGeneration,
    Qwen2_5_VLForConditionalGeneration,
    Trainer,
    TrainerCallback,
    is_wandb_available,
)
from transformers.integrations.deepspeed import is_deepspeed_zero3_enabled
from transformers.utils import is_peft_available

from trl.data_utils import apply_chat_template, is_conversational, maybe_apply_chat_template
from trl.models import create_reference_model, prepare_deepspeed, unwrap_model_for_generation
from trl.trainer.grpo_config import GRPOConfig
from trl.trainer.utils import generate_model_card, get_comet_experiment_url
import torch.distributed as dist
import copy
from qwen_vl_utils import process_vision_info

if is_peft_available():
    from peft import PeftConfig, get_peft_model

if is_wandb_available():
    import wandb

import torch
from torch.nn.utils.rnn import pad_sequence
from contextlib import nullcontext

# What we call a reward function is a callable that takes a list of prompts and completions and returns a list of
# rewards. When it's a string, it's a model ID, so it's loaded as a pretrained model.
RewardFunc = Union[str, PreTrainedModel, Callable[[list, list], list[float]]]

DEFAULT_VIDEO_MAX_PIXELS = 768 * 28 * 28
DEFAULT_VIDEO_MIN_PIXELS = 16 * 28 * 28
DEFAULT_INFER_MAX_NEW_TOKENS = 1024
_STANDALONE_OPT_RE = re.compile(r'(?<![A-Za-z0-9])([A-D])(?![A-Za-z0-9])', re.I)


class PostUpdateInferenceCallback(TrainerCallback):
    def __init__(self, trainer):
        self.trainer = trainer

    def on_step_end(self, args, state, control, model=None, **kwargs):
        self.trainer.run_post_update_inference(model)
        return control


SYSTEM_PROMPT = "You are an advanced anomaly detector assigned to analyze a video. "


QUESTION_TEMPLATE = """
            You are a helpful AI assistant performing video anomaly detection reasoning. 
            Below is a video description and a reasoning process. Based on this, answer the following multiple-choice question correctly.

            ### Question:
            [QUESTION]

            ### Options:
            "A. "[OPTION_A]
            "B. "[OPTION_B]
            "C. "[OPTION_C]
            "D. "[OPTION_D]

            When the user asks a question, you should first carefully reason through the problem internally, and then present the final option.
            The expected format is: <think> your detailed reasoning process here </think><answer> Please only output one option letter! (A, B, C, or D) </answer>. 
            """

QUESTION_TEMPLATE_WO_THINK = """
            You are a helpful AI assistant performing video anomaly detection reasoning. 
            Below is a video description and a reasoning process. Based on this, answer the following multiple-choice question correctly.

            ### Question:
            [QUESTION]

            ### Options:
            "A. "[OPTION_A]
            "B. "[OPTION_B]
            "C. "[OPTION_C]
            "D. "[OPTION_D]

            Please only output one option letter (A, B, C, or D). Please do not output other words.
"""

class RankGroupedVideoBatchSampler(torch.utils.data.Sampler):
    """
    Yield rank-local batches that keep the two QA samples for one video together.

    DDP/Accelerate may shard ordinary sample indices across ranks, so relying on
    adjacent dataset rows is not enough for this training flow.
    """

    def __init__(self, dataset, group_size, batch_size, rank, world_size, drop_last=False):
        self.dataset = dataset
        self.group_size = group_size
        self.batch_size = max(group_size, (batch_size // group_size) * group_size)
        self.rank = rank
        self.world_size = world_size
        self.drop_last = drop_last
        self.num_padded_groups = 0
        self.batches = self._build_batches()

    def _build_batches(self):
        grouped_indices = []
        pending_by_video = defaultdict(list)

        for idx in range(len(self.dataset)):
            example = self.dataset[idx]
            video_name = os.path.basename(example.get("video_path", "unknown"))
            pending = pending_by_video[video_name]
            pending.append(idx)

            while len(pending) >= self.group_size:
                grouped_indices.append(pending[: self.group_size])
                del pending[: self.group_size]

        if grouped_indices and self.world_size > 1:
            remainder = len(grouped_indices) % self.world_size
            if remainder:
                self.num_padded_groups = self.world_size - remainder
                padding_groups = [list(group) for group in grouped_indices[: self.num_padded_groups]]
                grouped_indices.extend(padding_groups)

        rank_groups = [
            group for group_idx, group in enumerate(grouped_indices)
            if group_idx % self.world_size == self.rank
        ]

        batches = []
        groups_per_batch = max(1, self.batch_size // self.group_size)
        for start in range(0, len(rank_groups), groups_per_batch):
            batch_groups = rank_groups[start: start + groups_per_batch]
            if self.drop_last and len(batch_groups) < groups_per_batch:
                continue
            batch = [idx for group in batch_groups for idx in group]
            batches.append(batch)

        return batches

    def __iter__(self):
        yield from self.batches

    def __len__(self):
        return len(self.batches)


class Qwen2VLGRPOTrainer_Video_QA(Trainer):

    def __init__(
        self,
        model: Union[str, PreTrainedModel],
        reward_funcs: Union[RewardFunc, list[RewardFunc]],
        args: GRPOConfig = None,
        train_dataset: Optional[Union[Dataset, IterableDataset]] = None,
        eval_dataset: Optional[Union[Dataset, IterableDataset, dict[str, Union[Dataset, IterableDataset]]]] = None,
        processing_class: Optional[PreTrainedTokenizerBase] = None,
        reward_processing_classes: Optional[Union[PreTrainedTokenizerBase, list[PreTrainedTokenizerBase]]] = None,
        callbacks: Optional[list[TrainerCallback]] = None,
        optimizers: tuple[Optional[torch.optim.Optimizer], Optional[torch.optim.lr_scheduler.LambdaLR]] = (None, None),
        peft_config: Optional["PeftConfig"] = None,
        max_pixels: Optional[int] = DEFAULT_VIDEO_MAX_PIXELS,
        min_pixels: Optional[int] = DEFAULT_VIDEO_MIN_PIXELS,
        attn_implementation: str = "flash_attention_2",
        use_think_prompt: bool = False,
        dataset_name: Optional[str] = None,
        project_root: Optional[str] = None,
        # attn_implementation: str = "sdpa",
    ):
        # Args
        if args is None:
            model_name = model if isinstance(model, str) else model.config._name_or_path
            model_name = model_name.split("/")[-1]
            args = GRPOConfig(f"{model_name}-GRPO")

        # Models
        # Trained model
        model_init_kwargs = args.model_init_kwargs or {}
        model_init_kwargs["attn_implementation"] = attn_implementation
        if model_init_kwargs.get("torch_dtype") is None:
            model_init_kwargs["torch_dtype"] = torch.bfloat16

        if isinstance(model, str):
            model_id = model
            torch_dtype = model_init_kwargs.get("torch_dtype")
            if isinstance(torch_dtype, torch.dtype) or torch_dtype == "auto" or torch_dtype is None:
                pass  # torch_dtype is already a torch.dtype or "auto" or None
            elif isinstance(torch_dtype, str):  # it's a str, but not "auto"
                torch_dtype = getattr(torch, torch_dtype)
                model_init_kwargs["torch_dtype"] = torch_dtype
            else:
                raise ValueError(
                    "Invalid `torch_dtype` passed to `GRPOConfig`. Expected either 'auto' or a string representing "
                    f"a `torch.dtype` (e.g., 'float32'), but got {torch_dtype}."
                )
            model_init_kwargs["use_cache"] = (
                False if args.gradient_checkpointing else model_init_kwargs.get("use_cache")
            )
            
            if "Qwen2-VL" in model_id:
                model = Qwen2VLForConditionalGeneration.from_pretrained(model, **model_init_kwargs)
            elif "Qwen2.5-VL" in model_id:
                model_init_kwargs.pop("use_cache", None) 
                model_init_kwargs.pop("use_sliding_window", None)
                # ------------------------------------------
                model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                    model,
                    **model_init_kwargs
                    )
            elif "Aria" in model_id:
                model_init_kwargs.pop("use_cache", None) 
                model = AriaForConditionalGeneration.from_pretrained(model, **model_init_kwargs)
            else:
                model = AutoModelForCausalLM.from_pretrained(model, **model_init_kwargs)
        else:
            model_id = model.config._name_or_path
            if args.model_init_kwargs is not None:
                raise ValueError(
                    "You passed `model_init_kwargs` to the `GRPOConfig`, but your model is already instantiated. "
                    "This argument can only be used when the `model` argument is a string."
                )

        if peft_config is not None:
            model = get_peft_model(model, peft_config)
            model.train()
            model.print_trainable_parameters()
        model.config.use_cache = False if args.gradient_checkpointing else bool(getattr(model.config, "use_cache", False))

        # Reference model
        if is_deepspeed_zero3_enabled():
            if "Qwen2-VL" in model_id:
                self.ref_model = Qwen2VLForConditionalGeneration.from_pretrained(model_id, **model_init_kwargs)
            if "Qwen2.5-VL" in model_id:
                model_init_kwargs.pop("use_cache", None)
                model_init_kwargs.pop("use_sliding_window", None)
                
                self.ref_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                    model_id,
                    **model_init_kwargs
                )
            elif "Aria" in model_id:
                self.ref_model = AriaForConditionalGeneration.from_pretrained(model_id, **model_init_kwargs)
            else:
                self.ref_model = AutoModelForCausalLM.from_pretrained(model_id, **model_init_kwargs)
        elif peft_config is None:
            # If PEFT configuration is not provided, create a reference model based on the initial model.
            self.ref_model = create_reference_model(model)
        else:
            # If PEFT is used, the reference model is not needed since the adapter can be disabled
            # to revert to the initial model.
            self.ref_model = None

        # Processing class
        if processing_class is None:
            if "Qwen2-VL" in model_id or "Qwen2.5-VL" in model_id or "Aria" in model_id:
                processing_class = AutoProcessor.from_pretrained(
                    model_id,
                    min_pixels=min_pixels,
                    max_pixels=max_pixels,
                )
                pad_token_id = processing_class.tokenizer.pad_token_id
                processing_class.pad_token_id = pad_token_id
                processing_class.eos_token_id = processing_class.tokenizer.eos_token_id
                if "Qwen" in model_id or "Qwen2.5-VL" in model_id:
                    processing_class.image_processor.max_pixels = max_pixels
                    processing_class.image_processor.min_pixels = min_pixels
            else:
                processing_class = AutoTokenizer.from_pretrained(model.config._name_or_path, padding_side="left")
                pad_token_id = processing_class.pad_token_id

        # Reward functions
        if not isinstance(reward_funcs, list):
            reward_funcs = [reward_funcs]
        for i, reward_func in enumerate(reward_funcs):
            if isinstance(reward_func, str):
                reward_funcs[i] = AutoModelForSequenceClassification.from_pretrained(
                    reward_func, num_labels=1, **model_init_kwargs
                )
        self.reward_funcs = reward_funcs

        # Reward processing class
        if reward_processing_classes is None:
            reward_processing_classes = [None] * len(reward_funcs)
        elif not isinstance(reward_processing_classes, list):
            reward_processing_classes = [reward_processing_classes]
        else:
            if len(reward_processing_classes) != len(reward_funcs):
                raise ValueError("The number of reward processing classes must match the number of reward functions.")

        for i, (reward_processing_class, reward_func) in enumerate(zip(reward_processing_classes, reward_funcs)):
            if isinstance(reward_func, PreTrainedModel):
                if reward_processing_class is None:
                    reward_processing_class = AutoTokenizer.from_pretrained(reward_func.config._name_or_path)
                if reward_processing_class.pad_token_id is None:
                    reward_processing_class.pad_token = reward_processing_class.eos_token
                # The reward model computes the reward for the latest non-padded token in the input sequence.
                # So it's important to set the pad token ID to the padding token ID of the processing class.
                reward_func.config.pad_token_id = reward_processing_class.pad_token_id
                reward_processing_classes[i] = reward_processing_class
        self.reward_processing_classes = reward_processing_classes

        # Data collator
        def data_collator(features):  # No data collation is needed in GRPO
            return features

        # Training arguments
        # self.max_prompt_length = args.max_prompt_length
        self.max_prompt_length = getattr(args, "max_prompt_len", 1024)
        self.max_completion_length = args.max_completion_length  # = |o_i| in the GRPO paper
        print(self.max_completion_length)
        self.num_generations = args.num_generations  # = G in the GRPO paper
        self.video_max_pixels = int(max_pixels)
        self.video_min_pixels = int(min_pixels)
        self.use_think_prompt = bool(use_think_prompt)
        self.question_template = QUESTION_TEMPLATE if self.use_think_prompt else QUESTION_TEMPLATE_WO_THINK
        self.prompt_mode = "wthink" if self.use_think_prompt else "wothink"
        self.dataset_name = self._sanitize_path_component(dataset_name or "unknown")
        self.project_root = Path(project_root).resolve() if project_root else Path(__file__).resolve().parents[3]
        self.consensus_debug_save_path = str(
            self.project_root/f"results/qa"
            / f"reward_entropy_consensus_anchor_rewrite_{self.dataset_name}_{self.prompt_mode}.jsonl"
        )
        self.post_train_infer_save_path = str(
            self.project_root/f"results/qa"
            / f"result_entropy_consensus_anchor_rewrite_{self.dataset_name}_{self.prompt_mode}.jsonl"
        )
        self.progress_save_path = str(
            self.project_root/f"results/qa"
            / f"progress_entropy_consensus_anchor_rewrite_{self.dataset_name}_{self.prompt_mode}.jsonl"
        )
        print(
            f"--- Prompt mode: {self.prompt_mode}; dataset={self.dataset_name}; "
            f"progress_log={self.progress_save_path}; "
            f"post_infer_log={self.post_train_infer_save_path}; "
            f"debug_log={self.consensus_debug_save_path}"
        )
        self.infer_max_new_tokens = min(
            DEFAULT_INFER_MAX_NEW_TOKENS,
            max(512, int(self.max_completion_length)),
        )
        self.generation_config = GenerationConfig(
            max_new_tokens=self.max_completion_length,
            do_sample=True,  
            temperature=1, # HACK
            num_return_sequences=self.num_generations,
            pad_token_id=pad_token_id,
        )
        self.beta = args.beta

        # Initialize the metrics
        self._metrics = defaultdict(list)

        super().__init__(
            model=model,
            args=args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=processing_class,
            callbacks=callbacks,
            optimizers=optimizers,
        )

        self.model_accepts_loss_kwargs = False

        if self.ref_model is not None:
            if self.is_deepspeed_enabled:
                self.ref_model = prepare_deepspeed(self.ref_model, self.accelerator)
            else:
                self.ref_model = self.accelerator.prepare_model(self.ref_model, evaluation_mode=True)

        for i, reward_func in enumerate(self.reward_funcs):
            if isinstance(reward_func, PreTrainedModel):
                self.reward_funcs[i] = self.accelerator.prepare_model(reward_func, evaluation_mode=True)

        self.train_flow_group_size = 1
        self._train_flow_pending_by_video = defaultdict(list)
        self._post_update_infer_pending_groups = []
        self._test_ecva_by_video_name = None
        self._last_compute_loss_skip_backward = False
        self.add_callback(PostUpdateInferenceCallback(self))

    def _set_signature_columns_if_needed(self):
        # If `self.args.remove_unused_columns` is True, non-signature columns are removed.
        # By default, this method sets `self._signature_columns` to the model's expected inputs.
        # In GRPOTrainer, we preprocess data, so using the model's signature columns doesn't work.
        # Instead, we set them to the columns expected by the `training_step` method, hence the override.
        if self._signature_columns is None:
            self._signature_columns = ["prompt"]

    @staticmethod
    def _sanitize_path_component(value):
        value = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value).strip()).strip("._-")
        return value.lower() if value else "unknown"

    def _get_train_sampler(self, train_dataset=None):
        train_dataset = train_dataset if train_dataset is not None else self.train_dataset
        if train_dataset is None:
            return None

        return torch.utils.data.SequentialSampler(train_dataset)

    def _get_rank_and_world_size(self):
        if dist.is_available() and dist.is_initialized():
            return dist.get_rank(), dist.get_world_size()

        rank = int(os.environ.get("RANK", "0"))
        world_size = int(os.environ.get("WORLD_SIZE", "1"))
        return rank, world_size

    def get_train_dataloader(self):
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")
        if isinstance(self.train_dataset, IterableDataset):
            return super().get_train_dataloader()

        rank, world_size = self._get_rank_and_world_size()
        requested_batch_size = self.args.per_device_train_batch_size
        if requested_batch_size % self.train_flow_group_size != 0:
            print(
                f"--- [Rank {rank}] per_device_train_batch_size={requested_batch_size} "
                f"is not divisible by train_flow_group_size={self.train_flow_group_size}; "
                f"batching with multiples of {self.train_flow_group_size} for this run."
            )

        batch_sampler = RankGroupedVideoBatchSampler(
            dataset=self.train_dataset,
            group_size=self.train_flow_group_size,
            batch_size=requested_batch_size,
            rank=rank,
            world_size=world_size,
            drop_last=self.args.dataloader_drop_last,
        )

        print(
            f"--- [Rank {rank}/{world_size}] Building same-video QA-group dataloader: "
            f"{len(batch_sampler)} batches, effective_batch_size={batch_sampler.batch_size}, "
            f"padded_groups={batch_sampler.num_padded_groups}"
        )

        return torch.utils.data.DataLoader(
            self.train_dataset,
            batch_sampler=batch_sampler,
            collate_fn=self.data_collator,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory,
            persistent_workers=self.args.dataloader_num_workers > 0,
        )


    # Get the per-token log probabilities for the completions for the model and the reference model
    def _get_per_token_logps(
        self,
        model,
        input_ids,
        attention_mask,
        pixel_values_videos,
        video_grid_thw,
        logits_to_keep_from=0,
        return_token_entropies=False,
    ):
        model_kwargs = {
            "attention_mask": attention_mask,
            "pixel_values_videos": pixel_values_videos,
            "video_grid_thw": video_grid_thw,
        }

        # Qwen2.5-VL can avoid materializing prompt logits. Keep the last
        # completion tokens plus one final next-token logit, then drop the
        # final logit below for normal next-token alignment.
        used_model_logits_to_keep = False
        if logits_to_keep_from:
            model_kwargs["logits_to_keep"] = input_ids.size(1) - logits_to_keep_from
            used_model_logits_to_keep = True

        try:
            logits = model(input_ids, **model_kwargs).logits  # (B, kept_L, V)
        except TypeError:
            model_kwargs.pop("logits_to_keep", None)
            used_model_logits_to_keep = False
            logits = model(input_ids, **model_kwargs).logits  # (B, L, V)

        logits = logits[:, :-1, :]  # (B, L-1, V), exclude the last logit: it corresponds to the next token pred
        input_ids = input_ids[:, 1:]  # (B, L-1), exclude the first input ID since we don't have logits for it
        if logits_to_keep_from:
            input_ids = input_ids[:, logits_to_keep_from:]
        if logits_to_keep_from and not used_model_logits_to_keep:
            logits = logits[:, logits_to_keep_from:, :]

        # Compute selected-token log probabilities in small sequence chunks.
        # A full log_softmax over (long_video_seq_len x vocab) can add several
        # GB on top of the model logits during GRPO.
        chunk_size = int(os.environ.get("GRPO_LOGPROB_CHUNK_SIZE", "32"))
        per_token_logps = []
        per_token_entropies = []
        for logits_row, input_ids_row in zip(logits, input_ids):
            row_logps = []
            row_entropies = []
            for start in range(0, logits_row.size(0), chunk_size):
                end = min(start + chunk_size, logits_row.size(0))
                chunk_log_probs = logits_row[start:end].log_softmax(dim=-1)
                chunk_token_logps = torch.gather(
                    chunk_log_probs,
                    dim=1,
                    index=input_ids_row[start:end].unsqueeze(1),
                ).squeeze(1)
                row_logps.append(chunk_token_logps)
                if return_token_entropies:
                    chunk_probs = chunk_log_probs.exp()
                    row_entropies.append(-(chunk_probs * chunk_log_probs).sum(dim=-1).detach())
            per_token_logps.append(torch.cat(row_logps, dim=0))
            if return_token_entropies:
                per_token_entropies.append(torch.cat(row_entropies, dim=0))
        del logits
        stacked_logps = torch.stack(per_token_logps)
        if return_token_entropies:
            return stacked_logps, torch.stack(per_token_entropies)
        return stacked_logps


    # Trainer "prepares" the inputs before calling `compute_loss`. It converts to tensor and move to device.
    # Since we preprocess the data in `compute_loss`, we need to override this method to skip this step.
    def _prepare_inputs(self, inputs: dict[str, Union[torch.Tensor, Any]]) -> dict[str, Union[torch.Tensor, Any]]:
        return inputs

    @staticmethod
    def _option_text(options, index):
        if options is None or len(options) <= index:
            return ""
        text = str(options[index]).strip()
        return re.sub(r"^[A-D]\.\s*", "", text)

    def make_conversation_video(self, example, question_override=None):
        problem = example["problem"]
        options = problem.get("options", [])
        question = question_override if question_override is not None else problem["question"]
        example_prompt = self.question_template.replace("[QUESTION]", question)
        example_prompt = example_prompt.replace("[OPTION_A]", self._option_text(options, 0))
        example_prompt = example_prompt.replace("[OPTION_B]", self._option_text(options, 1))
        example_prompt = example_prompt.replace("[OPTION_C]", self._option_text(options, 2))
        example_prompt = example_prompt.replace("[OPTION_D]", self._option_text(options, 3))
        # print(example_prompt)
        return [
                # {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": [
                        {"type": "video", 
                        "video": example["video_path"], 
                        "max_pixels": self.video_max_pixels,
                        "min_pixels": self.video_min_pixels,
                        },
                        {"type": "text", "text": example_prompt},
                    ]
                },
            ]

    def extract_answer(self, content):
        if content is None:
            return None

        match = re.search(r'<answer>\s*([A-D])\s*</answer>', content, re.I)
        if match:
            return match.group(1).upper()

        match = re.search(r'<\s*([A-D])\s*/?\s*>', content, re.I)
        if match:
            return match.group(1).upper()

        parts = re.split(r'</think>', content, flags=re.I)
        tail = parts[-1] if len(parts) > 1 else ''
        opts_in_tail = _STANDALONE_OPT_RE.findall(tail)
        if opts_in_tail:
            return opts_in_tail[-1].upper()

        opts_all = _STANDALONE_OPT_RE.findall(content)
        if opts_all:
            return opts_all[-1].upper()

        return None

    def _build_prompt_inputs(self, inputs, prompts_text):
        video_inputs = [x["video_inputs"] for x in inputs][0]
        fps_inputs = [x["video_kwargs"]["fps"] for x in inputs][0]
        prompt_inputs = self.processing_class(
            text=[prompts_text[0]],
            images=None,
            videos=[video_inputs[0]],
            fps=[fps_inputs[0]],
            padding=True,
            return_tensors="pt",
            padding_side="left",
            add_special_tokens=False,
            max_pixels=self.video_max_pixels,
        )
        return super()._prepare_inputs(prompt_inputs)

    def _generate_rollouts(self, model, prompt_inputs):
        prompt_ids = prompt_inputs["input_ids"]
        prompt_mask = prompt_inputs["attention_mask"]
        with unwrap_model_for_generation(model, self.accelerator) as unwrapped_model:
            prompt_completion_ids = unwrapped_model.generate(
                **prompt_inputs,
                generation_config=self.generation_config,
            )
        prompt_length = prompt_ids.size(1)
        completion_ids = prompt_completion_ids[:, prompt_length:]
        prompt_mask = prompt_mask.repeat_interleave(self.num_generations, dim=0)
        return prompt_completion_ids, completion_ids, prompt_mask, prompt_length

    def _get_majority_answer(self, extracted_answers):
        valid_votes = [answer for answer in extracted_answers if answer is not None]
        if not valid_votes:
            return None

        counts = Counter(valid_votes)
        answer, count = counts.most_common(1)[0]
        if count > (self.num_generations / 2.0):
            return answer
        return None

    def _sync_oom_flag(self, local_oom):
        oom_tensor = torch.tensor(
            1 if local_oom else 0,
            dtype=torch.int,
            device=self.accelerator.device,
        )
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(oom_tensor, op=dist.ReduceOp.MAX)
        return bool(oom_tensor.item())

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):

        import os
        video_names = [os.path.basename(x.get("video_path", "unknown")) for x in inputs]
        
        local_rank = os.environ.get("LOCAL_RANK", "0")
        
        print(f"--- [GPU {local_rank}] Step: {self.state.global_step} | Processing Videos: {video_names}")
        self._last_compute_loss_skip_backward = False

        if return_outputs:
            raise ValueError("The GRPOTrainer does not support returning outputs")

        prompts = [self.make_conversation_video(example) for example in inputs]
        prompts_text = [self.processing_class.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True) for prompt in prompts]
        
        prompt_inputs = self._build_prompt_inputs(inputs, prompts_text)
        prompt_completion_ids, completion_ids, prompt_mask, prompt_length = self._generate_rollouts(model, prompt_inputs)

        is_eos = completion_ids == self.processing_class.eos_token_id
        device = self.accelerator.device
        eos_idx = torch.full((is_eos.size(0),), is_eos.size(1), dtype=torch.long, device=device)
        eos_idx[is_eos.any(dim=1)] = is_eos.int().argmax(dim=1)[is_eos.any(dim=1)]
        sequence_indices = torch.arange(is_eos.size(1), device=device).expand(is_eos.size(0), -1)
        completion_mask = (sequence_indices <= eos_idx.unsqueeze(1)).int()

        completions = self.processing_class.batch_decode(completion_ids, skip_special_tokens=True)

        extracted_answers = [self.extract_answer(c) for c in completions]
        original_majority_answer = self._get_majority_answer(extracted_answers)
        print(f"--- [GPU {local_rank}] Original answers: {extracted_answers}")

        rewritten_question = inputs[0].get("problem", {}).get("rewritten_question", "")
        rewritten_question = rewritten_question.strip() if isinstance(rewritten_question, str) else ""
        rewritten_completions = []
        rewritten_answers = []
        rewritten_majority_answer = None

        if rewritten_question:
            rewritten_prompts = [
                self.make_conversation_video(example, question_override=example.get("problem", {}).get("rewritten_question", ""))
                for example in inputs
            ]
            rewritten_prompts_text = [
                self.processing_class.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True)
                for prompt in rewritten_prompts
            ]
            rewritten_prompt_inputs = self._build_prompt_inputs(inputs, rewritten_prompts_text)
            _, rewritten_completion_ids, _, _ = self._generate_rollouts(model, rewritten_prompt_inputs)
            rewritten_completions = self.processing_class.batch_decode(
                rewritten_completion_ids,
                skip_special_tokens=True,
            )
            rewritten_answers = [self.extract_answer(completion) for completion in rewritten_completions]
            rewritten_majority_answer = self._get_majority_answer(rewritten_answers)
            print(f"--- [GPU {local_rank}] Rewritten answers: {rewritten_answers}")

        save_path = self.progress_save_path
        for example in inputs:
            correct_answer = example.get("solution", {}).get("answer", "unknown")
        log_entry = {
            "step": self.state.global_step,
            "rank": local_rank,
            "video": video_names[0],
            "correct_answer": correct_answer,
            "answers": extracted_answers,
            "original_majority_answer": original_majority_answer,
            "rewritten_question": rewritten_question,
            "rewritten_answers": rewritten_answers,
            "rewritten_majority_answer": rewritten_majority_answer,
        }

        try:
            with open(save_path, "a", encoding="utf-8") as f:
                fcntl.flock(f, fcntl.LOCK_EX)
                f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")
                fcntl.flock(f, fcntl.LOCK_UN)
        except Exception as e:
            print(f"Warning: Failed to save to {save_path}: {e}")
        # ------------------------
        skipped_metric_key = None
        skip_reason = None
        if original_majority_answer is None:
            skipped_metric_key = "skipped_no_majority"
            skip_reason = "the original question has no majority answer"
        elif not rewritten_question:
            skipped_metric_key = "skipped_missing_rewritten_question"
            skip_reason = "missing rewritten question"
        elif rewritten_majority_answer is None:
            skipped_metric_key = "skipped_rewritten_no_majority"
            skip_reason = "the rewritten question has no majority answer"
        elif original_majority_answer != rewritten_majority_answer:
            skipped_metric_key = "skipped_rewritten_disagreement"
            skip_reason = (
                f"original majority answer {original_majority_answer} differs from "
                f"rewritten majority answer {rewritten_majority_answer}"
            )

        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)

        pixel_values_videos = prompt_inputs["pixel_values_videos"].repeat(self.num_generations, 1)
        video_grid_thw = prompt_inputs["video_grid_thw"].repeat_interleave(self.num_generations, dim=0)
        # print(f"--- [GPU {local_rank}] Step: {self.state.global_step} | Start original logprob/KL path")

        per_token_logps, per_token_entropies = self._get_per_token_logps(
            model,
            prompt_completion_ids,
            attention_mask,
            pixel_values_videos,
            video_grid_thw,
            logits_to_keep_from=prompt_length - 1,
            return_token_entropies=True,
        )

        entropy_mask = (
            sequence_indices < eos_idx.unsqueeze(1)
        ).to(per_token_entropies.dtype)

        mean_token_entropies = (
            (per_token_entropies * entropy_mask).sum(dim=1)
            / entropy_mask.sum(dim=1).clamp_min(1)
        ).detach().cpu().tolist()

        with torch.inference_mode():
            if self.ref_model is not None:
                ref_per_token_logps = self._get_per_token_logps(
                    self.ref_model,
                    prompt_completion_ids,
                    attention_mask,
                    pixel_values_videos,
                    video_grid_thw,
                    logits_to_keep_from=prompt_length - 1,
                )
            else:
                with self.accelerator.unwrap_model(model).disable_adapter():
                    ref_per_token_logps = self._get_per_token_logps(
                        model,
                        prompt_completion_ids,
                        attention_mask,
                        pixel_values_videos,
                        video_grid_thw,
                        logits_to_keep_from=prompt_length - 1,
                    )

        per_token_kl = torch.exp(ref_per_token_logps - per_token_logps) - (ref_per_token_logps - per_token_logps) - 1
        
        prompts_for_reward = [prompt for prompt in prompts for _ in range(self.num_generations)]
        
        rewards_per_func = torch.zeros(len(prompts_for_reward), len(self.reward_funcs), device=device)
        

        for i, (reward_func, reward_processing_class) in enumerate(zip(self.reward_funcs, self.reward_processing_classes)):
            if isinstance(reward_func, PreTrainedModel):
                if is_conversational(inputs[0]):
                    messages = [{"messages": p + c} for p, c in zip(prompts, completions)]
                    texts = [apply_chat_template(x, reward_processing_class)["text"] for x in messages]
                else:
                    texts = [p + c for p, c in zip(prompts, completions)]
                reward_inputs = reward_processing_class(
                    texts, return_tensors="pt", padding=True, padding_side="right", add_special_tokens=False
                )
                reward_inputs = super()._prepare_inputs(reward_inputs)
                with torch.inference_mode():
                    rewards_per_func[:, i] = reward_func(**reward_inputs).logits[:, 0]  # Shape (B*G,)
 
            else:
                reward_kwargs = {key: [] for key in inputs[0].keys() if key not in ["prompt", "completion"]}
                for key in reward_kwargs:
                    for example in inputs:
                        reward_kwargs[key].extend([example[key]] * self.num_generations)
                consensus_debug_records = []
                reward_kwargs["global_step"] = self.state.global_step
                reward_kwargs["mean_token_entropies"] = mean_token_entropies
                reward_kwargs["consensus_debug_records"] = consensus_debug_records
                
                output_reward_func = reward_func(prompts=prompts_for_reward, completions=completions, **reward_kwargs)
                rewards_per_func[:, i] = torch.tensor(output_reward_func, dtype=torch.float32, device=device)

                if consensus_debug_records:
                    reward_func_name = reward_func.__name__ if hasattr(reward_func, "__name__") else str(reward_func)
                    consensus_debug_save_path = os.environ.get(
                        "GRPO_CONSENSUS_DEBUG_SAVE_PATH",
                        self.consensus_debug_save_path,
                    )
                    for consensus_debug in consensus_debug_records:
                        consensus_debug_entry = {
                            "step": self.state.global_step,
                            "rank": local_rank,
                            "reward_func": reward_func_name,
                            "videos": video_names,
                            "video": video_names[0] if video_names else "unknown",
                            "correct_answer": correct_answer,
                            "extracted_answers": extracted_answers,
                            "num_generations": self.num_generations,
                            "completion_token_lengths": completion_mask.sum(dim=1).detach().cpu().tolist(),
                            "reward_values": [float(value) for value in output_reward_func],
                            **consensus_debug,
                        }
                        try:
                            with open(consensus_debug_save_path, "a", encoding="utf-8") as f:
                                fcntl.flock(f, fcntl.LOCK_EX)
                                f.write(json.dumps(consensus_debug_entry, ensure_ascii=False) + "\n")
                                fcntl.flock(f, fcntl.LOCK_UN)
                        except Exception as e:
                            print(f"Warning: Failed to save consensus debug to {consensus_debug_save_path}: {e}")


        rewards = rewards_per_func.sum(dim=1)
        virtual_anchor_reward = float(os.environ.get("GRPO_VIRTUAL_ANCHOR_REWARD", "0.0"))
        virtual_adv_scale = float(os.environ.get("GRPO_VIRTUAL_ADV_SCALE", "1.0"))
        reward_equal_atol = float(os.environ.get("GRPO_VIRTUAL_REWARD_ATOL", "1e-6"))
        reward_equal_rtol = float(os.environ.get("GRPO_VIRTUAL_REWARD_RTOL", "1e-5"))

        grouped_rewards = rewards.view(-1, self.num_generations)
        group_advantages = []
        group_means = []
        group_stds = []
        virtual_anchor_flags = []

        for group_idx, group_rewards in enumerate(grouped_rewards):
            start = group_idx * self.num_generations
            end = start + self.num_generations
            group_answers = extracted_answers[start:end]

            same_valid_answer = (
                len(group_answers) == self.num_generations
                and all(answer is not None for answer in group_answers)
                and len(set(group_answers)) == 1
            )

            same_real_reward = torch.allclose(
                group_rewards,
                group_rewards[0].expand_as(group_rewards),
                atol=reward_equal_atol,
                rtol=reward_equal_rtol,
            )

            if same_valid_answer and same_real_reward:
                observed_answer = group_answers[0]
                virtual_answer = "A" if observed_answer == "B" else "B"
                virtual_reward = group_rewards.new_tensor([virtual_anchor_reward])
                augmented_rewards = torch.cat([group_rewards, virtual_reward], dim=0)

                group_mean = augmented_rewards.mean()
                group_std = augmented_rewards.std()
                group_advantage = (group_rewards - group_mean) / (group_std + 1e-4)

                group_advantage = group_advantage * virtual_adv_scale
                virtual_anchor_flags.append(1.0)

                print(
                    f"--- [GPU {local_rank}] Step: {self.state.global_step} | "
                    f"virtual counterexample: {observed_answer * self.num_generations} + "
                    f"{virtual_answer}(reward={virtual_anchor_reward:.4f}), "
                    f"real_adv={group_advantage.detach().cpu().tolist()}"
                )
            else:
                group_mean = group_rewards.mean()
                group_std = group_rewards.std()
                group_advantage = (group_rewards - group_mean) / (group_std + 1e-4)
                virtual_anchor_flags.append(0.0)

            group_advantages.append(group_advantage)
            group_means.append(group_mean)
            group_stds.append(group_std)

        advantages = torch.cat(group_advantages, dim=0)
        mean_grouped_rewards = torch.stack(group_means).repeat_interleave(self.num_generations)
        std_grouped_rewards = torch.stack(group_stds).repeat_interleave(self.num_generations)

        any_oom = self._sync_oom_flag(False)

        metric_virtual_anchor_flags = (
            [0.0] * len(virtual_anchor_flags)
            if skipped_metric_key is not None or any_oom
            else virtual_anchor_flags
        )
        virtual_anchor_tensor = torch.tensor(
            metric_virtual_anchor_flags, dtype=torch.float32, device=device
        )
        self._metrics["virtual_anchor_rate"].append(
            self.accelerator.gather_for_metrics(virtual_anchor_tensor).float().mean().item()
        )

        per_token_loss = torch.exp(per_token_logps - per_token_logps.detach()) * advantages.unsqueeze(1)
        per_token_loss = -(per_token_loss - self.beta * per_token_kl)
        loss = ((per_token_loss * completion_mask).sum(dim=1) / completion_mask.sum(dim=1)).mean()
        # print(f"--- [GPU {local_rank}] Step: {self.state.global_step} | Computed loss")

        if any_oom:
            loss = per_token_logps.sum() * 0.0

        if skipped_metric_key is not None:
            print(
                f"--- [GPU {local_rank}] Step: {self.state.global_step} | "
                f"{skip_reason}; skipping gradient update ---"
            )
            loss = per_token_logps.sum() * 0.0
            self._metrics[skipped_metric_key].append(1.0)

        self._metrics["completion_length"].append(self.accelerator.gather_for_metrics(completion_mask.sum(1)).float().mean().item())
        
        if skipped_metric_key is not None or any_oom:
            metric_rewards_per_func = torch.zeros_like(rewards_per_func)
            metric_rewards = torch.zeros_like(rewards)
            metric_reward_std = torch.zeros_like(std_grouped_rewards)
        else:
            metric_rewards_per_func = rewards_per_func
            metric_rewards = rewards
            metric_reward_std = std_grouped_rewards

        reward_per_func = self.accelerator.gather_for_metrics(metric_rewards_per_func).mean(0)
        for i, reward_func in enumerate(self.reward_funcs):
            reward_func_name = reward_func.__name__ if not isinstance(reward_func, PreTrainedModel) else "RM"
            self._metrics[f"rewards/{reward_func_name}"].append(reward_per_func[i].item())

        self._metrics["reward"].append(self.accelerator.gather_for_metrics(metric_rewards).mean().item())
        self._metrics["reward_std"].append(self.accelerator.gather_for_metrics(metric_reward_std).mean().item())
        
        mean_kl = ((per_token_kl * completion_mask).sum(dim=1) / completion_mask.sum(dim=1)).mean()
        metric_kl = torch.zeros_like(mean_kl) if skipped_metric_key is not None or any_oom else mean_kl
        self._metrics["kl"].append(self.accelerator.gather_for_metrics(metric_kl).mean().item())
        # print(f"--- [GPU {local_rank}] Step: {self.state.global_step} | Finished metric gather")

        return loss

    def _training_step_with_oom_sync(self, model, inputs, num_items_in_batch=None):
        self._last_compute_loss_skip_backward = False
        model.train()
        inputs = self._prepare_inputs(inputs)

        try:
            with self.compute_loss_context_manager():
                loss = self.compute_loss(
                    model,
                    inputs,
                    num_items_in_batch=num_items_in_batch,
                )
        except RuntimeError as error:
            if not self._is_oom_error(error):
                raise
            loss = self._record_oom_skip(model, inputs[0], error)

        if getattr(self.args, "n_gpu", 1) > 1:
            loss = loss.mean()

        if self._last_compute_loss_skip_backward:
            return loss.detach()

        if not getattr(self, "model_accepts_loss_kwargs", False):
            loss = loss / self.args.gradient_accumulation_steps

        self.accelerator.backward(loss)
        return loss.detach()

    def training_step(self, model, inputs, num_items_in_batch=None):
        ready_groups = []
        for example in inputs:
            video_name = os.path.basename(example.get("video_path", "unknown"))
            pending_inputs = self._train_flow_pending_by_video[video_name]
            pending_inputs.append(example)

            if len(pending_inputs) >= self.train_flow_group_size:
                ready_groups.append(pending_inputs[: self.train_flow_group_size])
                del pending_inputs[: self.train_flow_group_size]

        group_losses = []
        for group_inputs in ready_groups:
            trained_group_inputs = []

            for train_input in group_inputs:
                train_video_name = train_input.get("video_name") or os.path.basename(train_input.get("video_path", "unknown"))
                train_question = train_input.get("problem", {}).get("question", "unknown")
                local_rank = os.environ.get("LOCAL_RANK", "0")
                print(
                    f"--- [GPU {local_rank}] Step: {self.state.global_step} | "
                    f"training sample: {os.path.basename(train_video_name)} | Question: {train_question}"
                )
                loss = self._training_step_with_oom_sync(model, [train_input], num_items_in_batch)
                if not self._last_compute_loss_skip_backward:
                    trained_group_inputs.append(train_input)
                group_losses.append(loss.detach())

            if trained_group_inputs:
                self._post_update_infer_pending_groups.append(trained_group_inputs)

        return torch.stack(group_losses).mean()

    def run_post_update_inference(self, model):
        if not self._post_update_infer_pending_groups:
            return

        if self.is_deepspeed_enabled:
            model = self.model_wrapped
        elif model is None:
            model = self.model_wrapped

        pending_groups = self._post_update_infer_pending_groups
        self._post_update_infer_pending_groups = []

        for group_inputs in pending_groups:
            self.post_train_infer(model, group_inputs)
            self.accelerator.wait_for_everyone()


    def _zero_loss_from_model(self, model):
        for param in model.parameters():
            if param.requires_grad and param.numel() > 0:
                return param.reshape(-1)[0] * 0.0
        return torch.tensor(0.0, device=self.accelerator.device, requires_grad=True)

    import torch

    def post_train_infer(self, model, inputs):

        save_file = self.post_train_infer_save_path
        self._inference_and_save(model, inputs, phase="post", save_file=save_file)


    def _inference_and_save(self, model, inputs, phase, save_file):
        """
        Reuse the pre/post-training inference flow and write results to the target JSONL file.
        """

        # Clear CPU and GPU memory.
        gc.collect()
        torch.cuda.empty_cache()
        
        model.eval()
        
        origin_uses_checkpointing = getattr(model, "is_gradient_checkpointing", False)
        if origin_uses_checkpointing:
            model.gradient_checkpointing_disable()

        with torch.inference_mode(): 
            for example in inputs:
                video_path = example.get("video_path", "unknown")
                problem = example.get("problem", {})
                print(
                    f"--- [Step {self.state.global_step-1}] post-update {phase} inference: "
                    f"{os.path.basename(video_path)} | Question: {problem.get('question', 'unknown')}"
                )
                correct_answer = example.get("solution", {}).get("answer", "unknown")

                prompts = [self.make_conversation_video(example)]
                texts = [self.processing_class.apply_chat_template(p, tokenize=False, add_generation_prompt=True) for p in prompts]
                
                video_inputs = example["video_inputs"]
                fps_inputs = example["video_kwargs"]["fps"]
                
                actual_video = video_inputs[0] if isinstance(video_inputs, list) else video_inputs
                actual_fps = fps_inputs[0] if isinstance(fps_inputs, list) else fps_inputs

                infer_inputs = self.processing_class(
                    text=texts,
                    videos=[actual_video],
                    fps=[actual_fps],
                    # fps = actual_fps,
                    padding=True,
                    return_tensors="pt",
                    max_pixels=self.video_max_pixels,
                ).to(model.device)

                unwrapped_model = self.accelerator.unwrap_model(model)

                unwrapped_model.config.use_cache = True
                
                output_ids = unwrapped_model.generate(
                    **infer_inputs,
                    max_new_tokens=self.infer_max_new_tokens,
                    do_sample=False,
                    use_cache=True, 
                    pad_token_id=self.processing_class.tokenizer.pad_token_id,
                    eos_token_id=self.processing_class.tokenizer.eos_token_id,
                )


                prompt_length = infer_inputs.input_ids.shape[1]
                response = self.processing_class.batch_decode(
                    output_ids[:, prompt_length:], 
                    skip_special_tokens=True
                )[0]

                log_entry = {
                    "step": self.state.global_step-1,
                    "video_name": os.path.basename(video_path),
                    "question": problem.get("question"),
                    "correct_answer": correct_answer,
                    "model_response": response.strip()
                }
                
                with open(save_file, "a", encoding="utf-8") as f:
                    f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")

                # print(f"\nResult: \n {response}... (Saved to JSONL)")


        unwrapped_model.config.use_cache = False
        if origin_uses_checkpointing:
            model.gradient_checkpointing_enable()
        
        torch.cuda.empty_cache()
        model.train()
        

    def log(self, logs: dict[str, float], start_time: Optional[float] = None) -> None:
        metrics = {key: sum(val) / len(val) for key, val in self._metrics.items()}  # average the metrics
        logs = {**logs, **metrics}
        if version.parse(transformers.__version__) >= version.parse("4.47.0.dev0"):
            super().log(logs, start_time)
        else:  # transformers<=4.46
            super().log(logs)
        self._metrics.clear()

    def create_model_card(
        self,
        model_name: Optional[str] = None,
        dataset_name: Optional[str] = None,
        tags: Union[str, list[str], None] = None,
    ):
        """
        Creates a draft of a model card using the information available to the `Trainer`.

        Args:
            model_name (`str` or `None`, *optional*, defaults to `None`):
                Name of the model.
            dataset_name (`str` or `None`, *optional*, defaults to `None`):
                Name of the dataset used for training.
            tags (`str`, `list[str]` or `None`, *optional*, defaults to `None`):
                Tags to be associated with the model card.
        """
        if not self.is_world_process_zero():
            return

        if hasattr(self.model.config, "_name_or_path") and not os.path.isdir(self.model.config._name_or_path):
            base_model = self.model.config._name_or_path
        else:
            base_model = None

        tags = tags or []
        if isinstance(tags, str):
            tags = [tags]

        if hasattr(self.model.config, "unsloth_version"):
            tags.append("unsloth")

        citation = textwrap.dedent(
            """\
            @article{zhihong2024deepseekmath,
                title        = {{DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models}},
                author       = {Zhihong Shao and Peiyi Wang and Qihao Zhu and Runxin Xu and Junxiao Song and Mingchuan Zhang and Y. K. Li and Y. Wu and Daya Guo},
                year         = 2024,
                eprint       = {arXiv:2402.03300},
            """
        )

        model_card = generate_model_card(
            base_model=base_model,
            model_name=model_name,
            hub_model_id=self.hub_model_id,
            dataset_name=dataset_name,
            tags=tags,
            wandb_url=wandb.run.get_url() if is_wandb_available() and wandb.run is not None else None,
            comet_url=get_comet_experiment_url(),
            trainer_name="GRPO",
            trainer_citation=citation,
            paper_title="DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models",
            paper_id="2402.03300",
        )

        model_card.save(os.path.join(self.args.output_dir, "README.md"))
