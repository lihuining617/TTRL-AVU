import os
import torch
from transformers import Qwen2_5_VLForConditionalGeneration, Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
import argparse
import csv
from tqdm import tqdm
import pandas as pd
os.environ["TOKENIZERS_PARALLELISM"] = "false"


FIXED_OPTION = "['Abuse', 'Agriculture', 'Arrest', 'Arson', 'Assault', 'Explosion', 'Fighting', 'Fire', 'Food Safety', 'Medical Abnormality', 'Natural Hazard', 'Normal', 'Object_falling', 'People_falling', 'Robbery', 'Shooting', 'Stealing', 'Traffic_accident', 'Vandalism', 'Water_incident']"

FIXED_QUESTION = "What type of anomaly appears in the video? If no anomaly appears, please answer 'Normal'."

THINK_QUESTION_TEMPLATE = """Answer the question: "[QUESTION]" according to the content of the video. Select one of the answer from :[OPTION]. When the user asks a question, you should first carefully reason through the problem internally, and then present the final option. The expected format is: <think> Your detailed reasoning process here </think><answer> Output the corresponding option here </answer> """

NO_THINK_QUESTION_TEMPLATE = """Answer the question: "[QUESTION]" according to the content of the video. Select one of the answer from :[OPTION]. The expected format is: <answer> Output the corresponding option here </answer> """


class QwenVL:
    def __init__(
        self, model_path,
        max_new_tokens = 1024,
        max_pixels = 768 * 28 * 28,
        min_pixels = 256 * 28 * 28,
    ):
        if "Qwen2.5" in model_path:
            self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                model_path,
                torch_dtype=torch.bfloat16,
                attn_implementation="sdpa",
                device_map="auto",
            )
        elif "Qwen2-VL" in model_path:
            self.model = Qwen2VLForConditionalGeneration.from_pretrained(
                model_path,
                torch_dtype=torch.bfloat16,
                attn_implementation="flash_attention_2",
                device_map="auto",
            )
        self.model.config.use_cache = True

        self.processor = AutoProcessor.from_pretrained(model_path, min_pixels=min_pixels, max_pixels=max_pixels)
        self.gen_config = {
            "max_new_tokens": max_new_tokens, 
            "use_cache": True
        }
        self.max_pixels = max_pixels
        self.min_pixels = min_pixels


    def _wrap_video(self, path: str):
        return {
            "type": "video",
            "video": path,
            "max_pixels": self.max_pixels,
            "min_pixels": self.min_pixels,
        }


    def parse_input(self, query=None, imgs=None, vid=None):
        # pure-text question
        if imgs is None and vid is None:
            return [{"role": "user", "content": query}]

        # multimodal question
        content = []

        if imgs is not None:
            if isinstance(imgs, str):
                imgs = [imgs]
            for img in imgs:
                content.append({"type": "image", "image": img})

        if vid is not None:
            content.append(self._wrap_video(vid))

        content.append({"type": "text", "text": query})
        return [{"role": "user", "content": content}]


    @torch.inference_mode()
    def chat(self, query=None, imgs=None, vid=None, history=None):
        if history is None:
            history = [{"role": "system", "content": "You are a helpful assistant."}]

        history.extend(self.parse_input(query, imgs, vid))

        prompt_text = self.processor.apply_chat_template(
            history,
            tokenize=False,
            add_generation_prompt=True,
        )

        image_inputs, video_inputs, video_kwargs = process_vision_info(
            [history], return_video_kwargs=True
        )
        fps_inputs = video_kwargs.get("fps")

        proc_kwargs = dict(
            text=[prompt_text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        # if fps_inputs:
        #     proc_kwargs["fps"] = fps_inputs
        fps_inputs = video_kwargs.get("fps")
        if fps_inputs:
            # If this is a list, use the first value; otherwise use it directly.
            proc_kwargs["fps"] = fps_inputs[0] if isinstance(fps_inputs, list) else fps_inputs

        inputs = self.processor(**proc_kwargs).to("cuda")
        # print("input_ids:", inputs.input_ids.shape)

        outputs = self.model.generate(**inputs, **self.gen_config)
        # print("output_ids:", outputs.shape)
        trim = [
            out_ids[len(in_ids) :]
            for in_ids, out_ids in zip(inputs.input_ids, outputs)
        ]
        response = self.processor.batch_decode(
            trim, skip_special_tokens=True, clean_up_tokenization_spaces=True
        )[0]

        history.append({"role": "assistant", "content": response})

        # tidy-up
        del inputs, outputs, trim
        torch.cuda.empty_cache()
        return response, history


def inference(chat_model, vid_path, think):
    history = None

    if think:
        prompt = THINK_QUESTION_TEMPLATE.replace("[QUESTION]", FIXED_QUESTION).replace("[OPTION]", FIXED_OPTION)
    else:
        prompt = NO_THINK_QUESTION_TEMPLATE.replace("[QUESTION]", FIXED_QUESTION).replace("[OPTION]", FIXED_OPTION)

    response, history = chat_model.chat(query=prompt, vid=vid_path, history=history)
    return response


def main(args):
    chat_model = QwenVL(model_path=args.model_path)

    videos_list = []
    existing_videos = set()

    # skip the videos that are already in the csv file
    if os.path.exists(args.output_csv):
        with open(args.output_csv, mode="r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                existing_videos.add(row["Video Name"])
    
    # read the video list
    df = pd.read_csv(args.test_gt_path)
    videos_list = df["Video Name"].tolist()

    for video in tqdm(videos_list):
        video_path = os.path.join(args.video_folder, video)
        if video in existing_videos:
            print(f"Skipping already processed video: {video}")
            continue

        response = None
        try:
            response = inference(chat_model, video_path, args.think)
        except Exception as e:
            print(f"Error processing video {video}: {e}")
            continue

        # process each video
        with open(args.output_csv, mode="a", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            if os.stat(args.output_csv).st_size == 0:
                writer.writerow(["Video Name", "CLS answer"])
            writer.writerow([str(video), str(response)])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Video inference with Qwen2VL.")
    parser.add_argument("--dataset", type=str, choices=["msad", "ucf", "ecva"])
    parser.add_argument("--model_path", type=str, required=True, help="Path to the model.")
    parser.add_argument("--video_folder", type=str, required=True, help="Path to the video folder.")
    parser.add_argument("--test_gt_path", type=str, required=True, help="Path to the test ground truth CSV file.")
    parser.add_argument("--output_csv", type=str, help="Path to the output CSV file.")
    parser.add_argument("--think", action="store_true", help="Whether to use the think process.")
    args = parser.parse_args()

    if args.output_csv is None:
        args.output_csv = f"./results/cls/{'_'.join(args.model_path.split('/')[1:])}/output_{args.dataset}_gd_{'w_think' if args.think else 'no_think'}.csv"
    os.makedirs(os.path.dirname(args.output_csv), exist_ok=True)
    main(args)
