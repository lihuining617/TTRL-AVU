import json
import os
import re

# Input and output paths
input_file = "TTRL-AVU/results/qa/result_xxx.jsonl" #replace with your actual training results file path
stats_file = input_file.strip(".jsonl") + "_evaluation.txt"
_STANDALONE_OPT_RE = re.compile(r'(?<![A-Za-z0-9])([A-D])(?![A-Za-z0-9])', re.I)
def extract_answer(content):
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

def run_evaluation_and_update():
    if not os.path.exists(input_file):
        print(f"Error: file not found {input_file}")
        return

    updated_data = []
    correct_count = 0
    total_count = 0

    # Read and process the original file.
    with open(input_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    for line in lines:
        if not line.strip():
            continue
        
        data = json.loads(line)
        
        # Evaluate entries that have not been evaluated yet.
        if "evaluation" not in data:
            model_ans = extract_answer(data.get("model_response", ""))
            is_correct = (model_ans == data.get("correct_answer"))
            data["evaluation"] = is_correct
        else:
            is_correct = data["evaluation"]
        
        updated_data.append(data)
        
        total_count += 1
        if is_correct:
            correct_count += 1

    # 1. Write results back to the original file so every line has an evaluation field.
    with open(input_file, 'w', encoding='utf-8') as f:
        for entry in updated_data:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    # 2. Save statistics to a txt file.
    os.makedirs(os.path.dirname(stats_file), exist_ok=True)
    with open(stats_file, 'w', encoding='utf-8') as f:
        f.write(f"Evaluation Statistics:\n")
        f.write(f"Total entries: {total_count}\n")
        f.write(f"Correct: {correct_count}\n")
        f.write(f"Accuracy: {(correct_count/total_count*100 if total_count > 0 else 0):.2f}%\n")

    print("Processing complete.")
    print(f"Data updated in the original file: {input_file}")
    print(f"Statistics saved to: {stats_file}")

if __name__ == "__main__":
    run_evaluation_and_update()
