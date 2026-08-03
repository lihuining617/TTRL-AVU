import os
import shutil
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import Qwen2VLForConditionalGeneration
from peft import PeftModel


def cp_files(file_names, src_dir, dst_dir):
    """
    Move files from src_dir to dst_dir.
    """
    if not os.path.exists(dst_dir):
        os.makedirs(dst_dir)

    for file_name in file_names:
        src_file = os.path.join(src_dir, file_name)
        dst_file = os.path.join(dst_dir, file_name)
        if os.path.exists(src_file):
            shutil.copy(src_file, dst_file)
            print(f"Copied {src_file} to {dst_file}.")
        else:
            print(f"{src_file} does not exist.")


# 1. load original Qwen2-VL base model
base_model_path = "./huggingface/Qwen2-VL-2B-Instruct"
base_model = Qwen2VLForConditionalGeneration.from_pretrained(
    base_model_path,
    device_map="auto",             
    trust_remote_code=True
)

# 2. load LoRA adapter
lora_adapter_path = "./checkpoints/run_sft_video_tag.sh_20250508_193839"
model = PeftModel.from_pretrained(
    base_model,
    lora_adapter_path,
)

# 3. merge LoRA adapter into base model
model = model.merge_and_unload()
print("Successfully merged LoRA adapter into base model.")

# 4. save the full merged model
save_path = "./huggingface/Qwen2-VL-2B-Instruct-SFT-TAG-test"
model.save_pretrained(save_path)

# copy essential files to the new directory
cp_files(
    [
        "added_tokens.json",
        "preprocessor_config.json",
        "special_tokens_map.json",
        "tokenizer_config.json",
        "tokenizer.json",
        "config.json",
        "vocab.json",
    ],
    base_model_path,
    save_path
)  

# (optional) save tokenizer
# tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)
# tokenizer.save_pretrained(save_path)

print(f"Full merged model saved to: {save_path}")
