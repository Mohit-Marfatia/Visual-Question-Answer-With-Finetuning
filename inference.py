import argparse
import pandas as pd
from PIL import Image
from tqdm import tqdm
import torch
from transformers import BlipProcessor, BlipForQuestionAnswering

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_dir', type=str, required=True, help='Path to image folder')
    parser.add_argument('--csv_path', type=str, required=True, help='Path to image-metadata CSV')
    args = parser.parse_args()

    # Load metadata CSV
    df = pd.read_csv(args.csv_path)

    # Load model and processor, move model to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BlipForQuestionAnswering.from_pretrained("ashutoshj01/blip-vqa-base-finetuned").to(device)
    processor = BlipProcessor.from_pretrained("ashutoshj01/blip-vqa-base-finetuned")
    model.eval()

    generated_answers = []
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        image_path = f"{args.image_dir}/{row['image_name']}"
        question = str(row['question'])
        try:
            image = Image.open(image_path).convert("RGB")
            encoding = processor(image, question, return_tensors="pt").to(device)
            with torch.no_grad():
                outputs = model.generate(**encoding)
                answer = processor.tokenizer.decode(outputs[0], skip_special_tokens=True)
        except Exception as e:
            answer = "error"
        answer = str(answer).lower()
        generated_answers.append(answer)

    df["generated_answer"] = generated_answers
    df.to_csv("results.csv", index=False)

if __name__ == "__main__":
    main()
