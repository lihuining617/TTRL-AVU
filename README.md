# TTRL-AVU

This repository contains the code and annotations for TTRL-AVU, a reinforcement-learning-based approach to video anomaly understanding. The current release focuses on multiple-choice video question answering over UCF-Crime, ECVA, and MSAD.

## Environment Setup

The environment has been tested on Linux with Python 3.10, PyTorch 2.6.0, and CUDA-enabled GPUs. A complete Conda specification is provided in `environment.yml`.

```bash
conda env create -f environment.yml
conda activate ttrl-avu
```

Creating the environment downloads PyTorch and CUDA-related packages and may take some time. Make sure that the installed NVIDIA driver is compatible with the CUDA runtime used by PyTorch. You can verify the installation with:

```bash
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
```

The default training configuration uses the Qwen2.5-VL checkpoint from Hugging Face. If the checkpoint is gated or downloaded from a private mirror, authenticate with Hugging Face before training.

## Dataset Preparation

Download the original videos from the official dataset pages:

- [UCF-Crime](https://xuange923.github.io/Surveillance-Video-Understanding)
- [ECVA](https://github.com/Dulpy/ECVA)
- [MSAD](https://github.com/Tom-roujiang/MSAD)

The QA annotations used by this repository are provided in `annotations/`. After downloading a dataset, place its videos in a directory of your choice. Video filenames are expected to match the `Video Name` column in the corresponding CSV annotation file.

An example layout is:

```text
TTRL-AVU/
├── annotations/
│   ├── test_ucf_crime_rewritten.csv
│   ├── test_ecva_rewritten.csv
│   └── test_msad_rewritten.csv
└── data/
    ├── ucf_crime/
    ├── ecva/
    └── msad/
```

The video directories are not required to be inside the repository. Set `--train_video_folder` to the actual location when configuring training.

Please follow the licenses and terms of use of the original datasets. This repository does not redistribute the source videos.

## Training

Before running training, edit `scripts/training/run_grpo_video_qa.sh`. At minimum, review and update the following settings:

- `CUDA_VISIBLE_DEVICES`: comma-separated GPU IDs visible to the job.
- `--nproc_per_node`: number of training processes; normally this must equal the number of visible GPUs.
- `--model_name_or_path`: Hugging Face model ID or local path to the base model checkpoint.
- `--train_data_path`: path to the selected CSV annotation file.
- `--train_video_folder`: directory containing the corresponding videos.
- `--dataset_name`: short dataset identifier used in result filenames, for example `ucf`, `ecva`, or `msad`.
- `--use_think_prompt`: set to `true` for `<think>...</think><answer>...</answer>` output or `false` for answer-only output.
- `--num_generations`: number of responses sampled for each prompt during GRPO.
- `--generation_batch_size`: total rollout batch size. Keep it compatible with `--num_generations` and the available GPU memory.
- `--per_device_train_batch_size`, `--gradient_accumulation_steps`, `--max_completion_length`, and the DeepSpeed configuration: adjust these according to available memory and the intended experiment.
- `--master_port`: change this if the default port is already in use.

The script creates checkpoints in `checkpoints/`, TensorBoard logs in `logs/`, and QA JSONL outputs in `results/qa/`. From the repository root, run:

```bash
sh scripts/training/run_grpo_video_qa.sh
```

Equivalently, from the training-script directory:

```bash
cd scripts/training
sh run_grpo_video_qa.sh
```

To inspect training metrics:

```bash
tensorboard --logdir logs
```

## Evaluation

Training writes a post-training inference file similar to:

```text
results/qa/result_entropy_consensus_anchor_rewrite_<dataset>_<prompt-mode>.jsonl
```

Before evaluation, open `src/evaluation/evaluation_qa.py` and set `input_file` to the JSONL file to evaluate. The evaluator extracts an A/B/C/D answer, adds an `evaluation` field to each JSONL record, and writes aggregate accuracy to a neighboring `*_evaluation.txt` file.

Run the evaluator from its directory:

```bash
cd src/evaluation
python evaluation_qa.py
```

Note that the evaluator updates the input JSONL file in place. Keep a copy if the original predictions must remain unchanged.

## Citation

If this repository is useful in your research, please cite the associated paper when its citation information becomes available. This codebase is adapted from [VAU-R1](https://github.com/GVCLab/VAU-R1); please also cite the original work:

```bibtex
@misc{zhu2025vaur1,
  title         = {VAU-R1: Advancing Video Anomaly Understanding via Reinforcement Fine-Tuning},
  author        = {Liyun Zhu and Qixiang Chen and Xi Shen and Xiaodong Cun},
  year          = {2025},
  eprint        = {2505.23504},
  archivePrefix = {arXiv},
  primaryClass  = {cs.CV},
  url           = {https://arxiv.org/abs/2505.23504}
}
```

Please also cite UCF-Crime, ECVA, and MSAD when using the corresponding datasets, following the citation instructions on their official pages.

## Acknowledgements

This codebase was developed by modifying [GVCLab/VAU-R1](https://github.com/GVCLab/VAU-R1). We sincerely thank the VAU-R1 authors for releasing their code and annotations. We also thank the creators and maintainers of UCF-Crime, ECVA, MSAD, Qwen2.5-VL, Hugging Face Transformers, TRL, and DeepSpeed.
