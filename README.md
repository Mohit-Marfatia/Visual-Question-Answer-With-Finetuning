# Visual Question Answering - Inference

This repository provides an inference pipeline using a fine-tuned BLIP VQA model.

Note :- if using command line it requires any hugging face account login. 

## 🔧 Installation

Install the required dependencies using:

```bash
pip install -r requirements.txt
python inference.py \
  --image_dir "/home/ashutosh/Downloads/inference-setup/inference-setup/data" \
  --csv_path "/home/ashutosh/Downloads/inference-setup/inference-setup/data/metadata.csv"
