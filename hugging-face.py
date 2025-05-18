import torch
from transformers import BlipProcessor, BlipForQuestionAnswering

adapter_weights = torch.load("D:/College/IIITB/CV/Final-Project/results/rank8/best_lora_model.pth",
    map_location=torch.device("cpu"))

model_name = "Salesforce/blip-vqa-base"
processor = BlipProcessor.from_pretrained(model_name)
base_model = BlipForQuestionAnswering.from_pretrained(model_name)

print("Attempting to load weights directly into the base model...")

filtered_weights = {k: v for k, v in adapter_weights.items() if k in base_model.state_dict()}
missing = set(base_model.state_dict().keys()) - set(filtered_weights.keys())
unexpected = set(filtered_weights.keys()) - set(base_model.state_dict().keys())

base_model.load_state_dict(filtered_weights, strict=False)
print("Direct loading completed with partial weights")


merged_model = base_model
merged_model.eval()
# Push to Hugging Face Hub
merged_model.push_to_hub("ashutoshj01/blip-vqa-base-finetune")
processor.push_to_hub("ashutoshj01/blip-vqa-base-finetune")