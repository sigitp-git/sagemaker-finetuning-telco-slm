# Tech Stack

## Language
- Python 3.10+

## Core ML Libraries
- `transformers` — model loading, tokenization, training args
- `peft` — LoRA/QLoRA via `LoraConfig`, `get_peft_model`
- `trl` — supervised fine-tuning via `SFTTrainer`
- `datasets` — loading JSONL training data
- `accelerate` — multi-GPU distributed training
- `bitsandbytes` — 4-bit quantization for QLoRA
- `scikit-learn` — evaluation metrics (F1, precision, recall)

## AWS Services
- Amazon SageMaker Training Jobs — managed fine-tuning compute
- Amazon SageMaker Endpoints — managed model serving
- Amazon Bedrock — frontier model inference (Claude, Nova Pro)
- Amazon S3 — dataset, adapter, and results storage
- Amazon EC2 (g6e.2xlarge / g6e.12xlarge) — alternative manual compute
- AWS Outposts / Local Zones — on-premise edge deployment

## Models
- `mistralai/Mistral-Nemo-Base-2407` (Ministral 3 14B) — QLoRA 4-bit
- `Qwen/Qwen3-14B` — QLoRA 4-bit
- `google/gemma-3-12b-pt` — BF16 LoRA

## Reporting
- HTML report — self-contained static file with metrics tables, generated from `results.json`
- JavaScript PPT — native `.pptx` file generated via [pptxgenjs](https://gitbrent.github.io/PptxGenJS/) (`src/report_ppt.js`, Node.js)
- Report scripts live in `src/report_html.py` (Python) and `src/report_ppt.js` (Node.js)
- Output goes to `reports/report.html` and `reports/presentation.pptx`

## Data Format
- JSONL (`.jsonl`) — one JSON object per line, streamed for training

## Important Version Pins
- `pytorch_version="2.1"` — required; torch 2.10+cu128 has a CUBLAS regression that breaks BF16/FP16 training
- `transformers_version="4.36"`

## Common Commands

```bash
# Install dependencies
pip install transformers peft trl datasets accelerate bitsandbytes scikit-learn

# Authenticate with Hugging Face (required for gated models)
huggingface-cli login

# Upload data to S3
aws s3 mb s3://your-telco-llm-bucket
aws s3 cp train.jsonl s3://your-telco-llm-bucket/data/train.jsonl
aws s3 cp test.jsonl  s3://your-telco-llm-bucket/data/test.jsonl

# Upload trained adapter to S3
aws s3 cp ./adapter-dir s3://your-telco-llm-bucket/adapters/model-name/ --recursive

# Download results
aws s3 cp s3://your-telco-llm-bucket/results/results.json results.json

# Verify GPU on EC2
nvidia-smi
python3 -c "import torch; print(torch.cuda.is_available())"
```
