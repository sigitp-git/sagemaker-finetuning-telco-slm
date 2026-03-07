# SageMaker Fine-Tuning: 14B SLMs for 3GPP Root Cause Analysis

Step-by-step guide for a fine-tuning benchmark that compares fine-tuned 14B SLMs against frontier foundation models on automated root cause analysis of 3GPP protocol logs in 5G SA core networks.

All steps use AWS managed services, with Amazon SageMaker Training Jobs as the primary compute option.

---

## How to Run This Benchmark

### 1. Provision Infrastructure

**AWS Services: Amazon SageMaker Training Jobs (recommended) or Amazon EC2**

**Primary: Amazon SageMaker Training Jobs (recommended)**

SageMaker Training Jobs is the managed option — no instance provisioning, no SSH, no manual teardown. You provide a training script and an S3 dataset path, specify the instance type, and SageMaker handles the rest: spins up the GPU instance, runs the job, saves artifacts to S3, and terminates the instance automatically.

Steps:
1. Create an IAM role for SageMaker with `AmazonSageMakerFullAccess`, `AmazonS3FullAccess`, and `AmazonBedrockFullAccess` policies.
2. Upload your training script and dataset to S3 (covered in Step 2).
3. Submit a Training Job using the SageMaker Python SDK:

```python
from sagemaker.huggingface import HuggingFace

estimator = HuggingFace(
    entry_point="train.py",                        # your fine-tuning script
    source_dir="./src",
    instance_type="ml.g5.2xlarge",                 # 1× A10G GPU, equivalent to g6e.2xlarge
    instance_count=1,
    role="arn:aws:iam::ACCOUNT_ID:role/SageMakerRole",
    transformers_version="4.46.1",
    pytorch_version="2.3.0",
    py_version="py311",
    hyperparameters={
        "model_id": "mistralai/Mistral-Nemo-Base-2407",
        "max_steps": 325,
        "use_4bit": True,
    }
)

estimator.fit({"train": "s3://your-telco-llm-bucket/data/train.jsonl"})
# Adapter saved automatically to S3 when job completes
```

4. For Qwen3-14B QLoRA (4-bit, multi-GPU), use `ml.g5.12xlarge` (4× A10G GPUs) instead.
5. Monitor job progress in the [SageMaker Console](https://console.aws.amazon.com/sagemaker) → **Training** → **Training jobs**.

> **Why do all three models use QLoRA or need careful memory management on A10G GPUs?**
>
> It comes down to memory requirements during training — not just weight storage.
>
> Mistral-Nemo-Base-2407 is a 12B model. At BF16, that's 12B × 2 bytes ≈ 24GB — exactly the A10G's VRAM limit. While the weights technically fit, there's zero headroom left for activations, gradients, or optimizer states during training. In practice, `model.to("cuda")` OOMs before training even starts. QLoRA compresses weights to 4-bit (~6GB), leaving ~18GB for training overhead.
>
> Qwen3-14B is larger (14B params) and uses QLoRA with 4-bit quantization via bitsandbytes. Even at 4-bit the activations and optimizer states during training are heavy:
>
> - Qwen3-14B has a larger architecture with more attention heads and a wider hidden dimension than Mistral-Nemo, so even at 4-bit the activations and optimizer states during training are heavier
> - QLoRA dequantizes weights to BF16 during the forward/backward pass for gradient computation, so peak memory spikes significantly beyond what the static 4-bit footprint suggests
> - 14B × 0.5 bytes (4-bit) ≈ 7GB for weights alone, but with activations, gradients, and optimizer states you can easily hit 60–80GB during training
> - The 4× A10G on `ml.g5.12xlarge` gives you 96GB total VRAM, which handles those spikes comfortably across devices via `accelerate`
>
> The counterintuitive result: a smaller model in 4-bit QLoRA fits on 1 GPU, while a larger model in 4-bit QLoRA still needs 4 GPUs because training-time memory pressure is dominated by activations and optimizer state, not just weight storage.

> Important: pin `pytorch_version="2.3.0"` (with `cu121`) in the estimator. `torch 2.10+cu128` has a CUBLAS regression that breaks all bf16/fp16 training. The original `pytorch_version="2.1"` / `transformers_version="4.36"` DLC predates Mistral-Nemo and cannot load it — use `transformers_version="4.46.1"` with `pytorch_version="2.3.0"` and `py_version="py311"`.

---

**Alternative: Amazon EC2 (manual, lower cost for iterative experimentation)**

If you prefer direct GPU access for interactive development or debugging, launch a GPU-backed EC2 instance manually.

Steps:
1. Open the [EC2 Console](https://console.aws.amazon.com/ec2) and click **Launch Instance**.
2. Search for the AMI: **Deep Learning OSS Nvidia Driver AMI GPU PyTorch** (Ubuntu). Pre-installed with CUDA, PyTorch, and common ML libraries.
3. Select instance type `g6e.2xlarge` (1× L40S, $1.86/hr). For Qwen3-14B QLoRA, use `g6e.12xlarge` (4× L4 GPUs).
4. Attach an EBS volume of at least **200GB** (gp3) for model weights, datasets, and checkpoints.
5. Assign an IAM role with `AmazonS3FullAccess` and `AmazonBedrockFullAccess`.
6. Connect via AWS Systems Manager Session Manager (no port 22 needed).

```bash
# Verify GPU after connecting
nvidia-smi
python3 -c "import torch; print(torch.cuda.is_available())"

# Pin PyTorch version
pip install torch==2.9.0 --index-url https://download.pytorch.org/whl/cu121
```

---

### 2. Prepare Synthetic Training Data

**AWS Services: Amazon Bedrock, Amazon S3**

Use a frontier model via Amazon Bedrock to generate the synthetic 3GPP log dataset. This avoids needing real operator data for the initial experiment.

Steps:
1. Enable model access in the [Bedrock Console](https://console.aws.amazon.com/bedrock) → **Model access** → enable Claude 4.6 Opus or Nova Pro.
2. Write a data generation script that calls the Bedrock API to produce labeled examples. Each example = a synthetic 3GPP signaling log + a ground-truth JSON with root cause error codes.

```python
import boto3, json

bedrock = boto3.client("bedrock-runtime", region_name="us-east-1")

prompt = """Generate a synthetic 3GPP NAS/NGAP/RRC signaling log for a 5G SA core
showing a UPF degradation cascade failure. Include sympathetic noise events
(heartbeat timeouts, keepalives). Output JSON: {"log": "...", "root_cause": [...]}"""

response = bedrock.invoke_model(
    modelId="anthropic.claude-opus-4-5",
    body=json.dumps({"messages": [{"role": "user", "content": prompt}],
                     "max_tokens": 2048, "anthropic_version": "bedrock-2023-05-31"})
)
```

3. Generate 1,300 training examples and 1,000 test examples across all 8 failure types:
   - `core_network_failure`, `authentication_failure`, `normal`, `handover_failure`
   - `congestion`, `qos_violation`, `transport_jitter`, `radio_failure`

4. Upload the datasets to **Amazon S3**:

```bash
aws s3 mb s3://your-telco-llm-bucket
aws s3 cp train.jsonl s3://your-telco-llm-bucket/data/train.jsonl
aws s3 cp test.jsonl  s3://your-telco-llm-bucket/data/test.jsonl
```

---

### 3. Fine-Tune the SLMs

**AWS Service: Amazon SageMaker Training Jobs or Amazon EC2**

Run LoRA/QLoRA fine-tuning using the [Hugging Face TRL](https://github.com/huggingface/trl) library and PEFT.

Steps:
1. Install dependencies:

```bash
pip install sagemaker
pip install transformers peft trl datasets accelerate bitsandbytes huggingface_hub scikit-learn
```

Validate all packages are installed correctly:

```python
import importlib.metadata as m
for pkg in ['sagemaker','transformers','peft','trl','datasets','accelerate','bitsandbytes','huggingface_hub','scikit-learn']:
    try:
        print(f'{pkg}: {m.version(pkg)}')
    except Exception as e:
        print(f'{pkg}: MISSING ({e})')
```

Expected output:
```
sagemaker: 3.5.0
transformers: 5.3.0
peft: 0.18.1
trl: 0.29.0
datasets: 4.6.1
accelerate: 1.13.0
bitsandbytes: 0.49.2
huggingface_hub: 1.6.0
scikit-learn: 1.8.0
```

Install the Hugging Face CLI (`hf`) using the standalone installer:

```bash
# Install (requires python3-venv; install it first if missing)
sudo apt install -y python3.11-venv
curl -LsSf https://hf.co/cli/install.sh | bash

# Make the CLI available in the current shell session
export PATH="/home/ubuntu/.local/bin:$PATH"

# Persist PATH across sessions
echo 'export PATH="/home/ubuntu/.local/bin:$PATH"' >> ~/.bashrc
```

Validate the CLI is working:

```bash
hf --version
# Expected: 1.6.0
```

> Note: The installer uses `hf` as the command name (not `huggingface-cli`). It installs into a dedicated venv at `~/.hf-cli` and symlinks the binary to `~/.local/bin/hf`.

2. Download the base model from Hugging Face (requires HF token for gated models):

```bash
hf auth login
# Models: mistralai/Mistral-Nemo-Base-2407, Qwen/Qwen3-14B, google/gemma-3-12b-pt
```

3. Submit the SageMaker Training Job using `submit_training.py` (repo root):

```bash
# Ministral 3 14B — QLoRA 4-bit on 1× A10G (ml.g5.2xlarge)
python submit_training.py \
  --role arn:aws:iam::ACCOUNT_ID:role/SageMakerRole \
  --bucket your-telco-llm-bucket \
  --model_id mistralai/Mistral-Nemo-Base-2407 \
  --max_steps 325

# Qwen3-14B — QLoRA 4-bit on 4× A10G (ml.g5.12xlarge)
python submit_training.py \
  --role arn:aws:iam::ACCOUNT_ID:role/SageMakerRole \
  --bucket your-telco-llm-bucket \
  --model_id Qwen/Qwen3-14B \
  --max_steps 325

# Gemma 3 12B — BF16 LoRA on 1× A10G (ml.g5.2xlarge)
# Note: Gemma 3 12B fits in BF16 on a single A10G because its architecture
# is more memory-efficient than Mistral-Nemo despite similar param count.
python submit_training.py \
  --role arn:aws:iam::ACCOUNT_ID:role/SageMakerRole \
  --bucket your-telco-llm-bucket \
  --model_id google/gemma-3-12b-pt \
  --max_steps 325
```

The script auto-selects the correct instance type and quantization mode per model. Add `--wait` to block and stream status until the job completes.

> Why `use_4bit` differs per model:
>
> - **Mistral-Nemo-Base-2407** (`use_4bit=True`): 12B params × 2 bytes (BF16) ≈ 24GB — exactly the A10G's 24GB VRAM limit. While weights technically fit, there's zero headroom for activations, gradients, or optimizer states. In practice, `model.to("cuda")` OOMs before training starts. QLoRA compresses weights to 4-bit (~6GB), leaving ~18GB for training overhead on a single A10G.
> - **Qwen3-14B** (`use_4bit=True`): 14B params × 2 bytes (BF16) ≈ 28GB — already over a single A10G's limit. QLoRA compresses weights to 4-bit (~7GB static), but training-time activations and optimizer states push peak memory to 60–80GB, requiring 4× A10G (`ml.g5.12xlarge`, 96GB total). Without 4-bit you'd need A100s.
> - **Gemma-3-12b-pt** (`use_4bit=False`): 12B in BF16 fits on one A10G. Google trained Gemma in BF16 natively with a more memory-efficient architecture, so keeping it in BF16 preserves numerical fidelity and avoids the dequantization overhead of QLoRA.
>
> The pattern: use BF16 LoRA only when the model fits in available VRAM with headroom for training. Use QLoRA when it doesn't. Quantization adds complexity (dequantization during forward/backward pass, potential precision loss) so it's only worth it when necessary.

Monitor training progress:

```bash
# One-shot status check
aws sagemaker describe-training-job \
  --training-job-name <job-name> \
  --query 'TrainingJobStatus' --output text

# Poll until completion (built into submit_training.py with --wait flag)
python submit_training.py --role ... --bucket ... --model_id ... --wait

# Stream CloudWatch logs
aws logs tail /aws/sagemaker/TrainingJobs \
  --log-stream-name-prefix <job-name> --follow

# Poll every 60s until terminal state (bash loop)
while true; do
  RESULT=$(aws sagemaker describe-training-job \
    --training-job-name <job-name> \
    --query '{Status:TrainingJobStatus, Elapsed:TrainingTimeInSeconds, Failure:FailureReason}' \
    --output json)
  echo "[$(date -u '+%H:%M:%S')] $RESULT"
  STATUS=$(echo $RESULT | python3 -c "import sys,json; print(json.load(sys.stdin)['Status'])")
  if [[ "$STATUS" == "Completed" || "$STATUS" == "Failed" || "$STATUS" == "Stopped" ]]; then
    echo "Job reached terminal state: $STATUS"
    break
  fi
  sleep 60
done
```

**Mistral-Nemo QLoRA 4-bit — Training Output (Completed)**

```
 98%|█████████▊| 320/325 [40:22<00:34,  6.85s/it]
 99%|█████████▉| 321/325 [40:29<00:27,  6.97s/it]
 99%|█████████▉| 322/325 [40:35<00:20,  6.84s/it]
 99%|█████████▉| 323/325 [40:41<00:13,  6.59s/it]
100%|█████████▉| 324/325 [40:49<00:06,  6.89s/it]
100%|██████████| 325/325 [40:56<00:00,  6.86s/it]
{'loss': 1.0355, 'grad_norm': 2.1321821212768555, 'learning_rate': 0.0, 'epoch': 2.01}
100%|██████████| 325/325 [40:57<00:00,  7.56s/it]
{'train_runtime': 2457.0749, 'train_samples_per_second': 1.058, 'train_steps_per_second': 0.132, 'train_loss': 1.3590671245868389, 'epoch': 2.01}
Adapter saved to /opt/ml/output/data/adapter
sagemaker-training-toolkit INFO  Reporting training SUCCESS
```

> **Training Summary — Mistral-Nemo QLoRA 4-bit**
>
> - **Job**: `telco-rca-mistral-nemo-base-2407-2026-03-07-18-19-25-833`
> - **Status**: Completed
> - **Steps**: 325/325, ~41 min training time, 2 epochs
> - **Final loss**: 1.035 (started ~2.0, dropped steadily through 1.35 → 1.09 → 0.98 → 1.03)
> - **Average train loss**: 1.359
> - **Adapter saved to**: `/opt/ml/output/data/adapter`
> - **Output uploaded to**: `s3://your-telco-llm-bucket/output/mistral-nemo-base-2407/`
>
**Mistral-Nemo QLoRA 4-bit — Training Metrics Log**

```bash
# Extract loss metrics from CloudWatch logs
aws logs tail /aws/sagemaker/TrainingJobs \
  --log-stream-name-prefix <job-name> \
  --since 3h 2>/dev/null \
  | grep "'loss'" \
  | grep -oP "'loss': [0-9.]+.*'epoch': [0-9.]+"
```

```
Step  25  | loss: 3.4207 | grad_norm: 3.0827 | lr: 1.997e-04 | epoch: 0.15
Step  50  | loss: 1.6251 | grad_norm: 2.5550 | lr: 1.944e-04 | epoch: 0.31
Step  75  | loss: 1.3590 | grad_norm: 2.6612 | lr: 1.830e-04 | epoch: 0.46
Step 100  | loss: 1.3540 | grad_norm: 1.8951 | lr: 1.663e-04 | epoch: 0.62
Step 125  | loss: 1.3426 | grad_norm: 2.0330 | lr: 1.452e-04 | epoch: 0.77
Step 150  | loss: 1.2488 | grad_norm: 1.8794 | lr: 1.213e-04 | epoch: 0.93
Step 175  | loss: 1.1902 | grad_norm: 1.8159 | lr: 9.592e-05 | epoch: 1.08
Step 200  | loss: 1.0485 | grad_norm: 2.3825 | lr: 7.085e-05 | epoch: 1.23
Step 225  | loss: 1.0046 | grad_norm: 1.8702 | lr: 4.766e-05 | epoch: 1.39
Step 250  | loss: 1.0953 | grad_norm: 2.4519 | lr: 2.786e-05 | epoch: 1.54
Step 275  | loss: 0.9816 | grad_norm: 1.7921 | lr: 1.273e-05 | epoch: 1.70
Step 300  | loss: 0.9620 | grad_norm: 2.2479 | lr: 3.234e-06 | epoch: 1.85
Step 325  | loss: 1.0355 | grad_norm: 2.1322 | lr: 0.000e+00 | epoch: 2.01
```

> **What does the loss curve mean?**
>
> The *training loss* measures how wrong the model's predictions are on each batch — lower is better. At the start of training, the model has never seen 3GPP logs, so its predictions are essentially random (loss ≈ 3.4). As it processes more batches, it learns the structure of the logs and the mapping to root cause codes, and the loss drops rapidly.
>
> The progression 3.42 → 1.63 → 1.36 → 1.05 → 0.96 → 1.04 shows a healthy convergence pattern:
> - **Epoch 1 (steps 1–162)**: Steep drop from 3.4 to ~1.2 — the model is learning the basic task structure (log format, output schema, common failure patterns).
> - **Epoch 2 (steps 163–325)**: Gradual refinement from ~1.2 to ~1.0 — the model is fine-tuning its understanding of subtle differences between failure types (e.g., distinguishing `congestion` from `qos_violation`).
> - **Final loss 1.035**: The slight uptick from 0.96 → 1.04 at the very end is normal — it's batch-level noise, not overfitting. The overall trend is clearly downward.
>
> The *average train loss* of 1.359 is the mean across all 325 steps. It's higher than the final loss because it includes the early high-loss steps when the model was still learning.
>
> ```
> Loss
> 3.5 │ ●
>     │
> 3.0 │
>     │
> 2.5 │
>     │
> 2.0 │
>     │
> 1.5 │    ●
>     │       ● ● ●
>     │              ●
>     │                 ●
> 1.0 │                    ● ●  ●  ● ● ●
>     │
> 0.5 │
>     │
> 0.0 └──────────────────────────────────────
>     0.0  0.3  0.5  0.8  1.0  1.2  1.5  1.7  2.0
>                        Epoch
>
> Mistral-Nemo QLoRA 4-bit — Loss Curve (325 steps, 2 epochs)
> ```
>
> The loss curve looks healthy — it converged nicely over 2 epochs. The adapter is ready for evaluation.

4. Save the LoRA adapter and upload to S3:

```bash
# Adapter is saved automatically by train.py to {output_dir}/adapter/
# SageMaker uploads it to s3://your-telco-llm-bucket/output/<model-slug>/ on job completion
# Download locally if needed:
aws s3 cp s3://your-telco-llm-bucket/output/mistral-nemo-base-2407/adapter/ \
  ./adapters/ministral/ --recursive
```

Training cost reference:

| Model | Method | GPU | Training Cost |
|-------|--------|-----|---------------|
| Ministral 3 14B | QLoRA 4-bit | 1× L40S | $2.57 |
| Qwen3-14B V5 | QLoRA 4-bit | 4× L4 | $2.42 |
| Gemma 3 12B | BF16 LoRA | 1× L40S | $3.44 |

---

### 4. Evaluate Frontier Models via Bedrock

**AWS Service: Amazon Bedrock**

Run Claude 4.6 Opus and Amazon Nova Pro against the 1,000-scenario test set using three prompt strategies per model.

Steps:
1. Ensure the IAM role has `AmazonBedrockFullAccess`.
2. Loop through the test set and call Bedrock for each scenario:

```python
import boto3, json

bedrock = boto3.client("bedrock-runtime", region_name="us-east-1")

def evaluate_bedrock(log_text, model_id, few_shot_examples=None, use_cot=False):
    messages = []
    if few_shot_examples:
        for ex in few_shot_examples:          # 5-shot
            messages.append({"role": "user", "content": ex["input"]})
            messages.append({"role": "assistant", "content": ex["output"]})
    cot_suffix = " Think step by step." if use_cot else ""
    messages.append({"role": "user", "content": log_text + cot_suffix})

    response = bedrock.invoke_model(
        modelId=model_id,   # "anthropic.claude-opus-4-5" or "amazon.nova-pro-v1:0"
        body=json.dumps({"messages": messages, "max_tokens": 1024,
                         "anthropic_version": "bedrock-2023-05-31"})
    )
    return json.loads(response["body"].read())
```

3. Run all three strategies (zero-shot, 5-shot, 5-shot + CoT) and record outputs for scoring.
4. Use the best-performing variant per model for the final comparison table.

---

### 5. Apply a Deterministic Post-Processing Filter

**AWS Service: Amazon EC2 or SageMaker (same environment as training)**

Before scoring any model output, apply the same noise-removal filter to all responses — both fine-tuned SLMs and Bedrock frontier models. This strips sympathetic noise events (heartbeat timeouts, keepalives, cascading consequential failures) from the predicted root cause list.

```python
SYMPATHETIC_CODES = {"HEARTBEAT_TIMEOUT", "KEEPALIVE_FAIL", "SECONDARY_ALARM", ...}

def filter_sympathetic_noise(predicted_codes: list) -> list:
    return [code for code in predicted_codes if code not in SYMPATHETIC_CODES]
```

Apply this to every model's output before computing any metric. This ensures the comparison is fair and no model is penalized or rewarded for how it handles noise.

---

### 6. Score with Consistent Metrics

**AWS Service: Amazon EC2 or SageMaker**

Compute all metrics against the ground-truth test set using the filtered model outputs:

```python
from sklearn.metrics import f1_score, precision_score, recall_score

def score(predictions, ground_truth):
    f1        = f1_score(ground_truth, predictions, average="micro")
    precision = precision_score(ground_truth, predictions, average="micro")
    recall    = recall_score(ground_truth, predictions, average="micro")
    exact_match = sum(p == g for p, g in zip(predictions, ground_truth)) / len(ground_truth)
    return {"f1": f1, "precision": precision, "recall": recall, "exact_match": exact_match}
```

Compute scores globally and per-scenario (across all 8 failure types) to produce the full ranking and scenario status matrix.

Store results back to S3 for reproducibility:

```bash
aws s3 cp results.json s3://your-telco-llm-bucket/results/results.json
```

---

### 7. Validate with Real Operator Data

**AWS Services: Amazon S3, AWS PrivateLink / VPC, Amazon SageMaker**

The above steps use synthetic data. For production validation with a real telco operator:

1. The operator uploads anonymized/sanitized 3GPP signaling logs to an **S3 bucket inside their own AWS account** (1,000–2,000 labeled examples with NOC-verified root causes).
2. Grant cross-account S3 access via an **S3 bucket policy** or use **AWS Resource Access Manager (RAM)** — the operator retains full data ownership and control.
3. Optionally, run the entire pipeline inside the operator's VPC using **AWS PrivateLink** so data never leaves their environment.
4. Re-run Steps 3–6 using the real dataset. Compare F1 scores against the synthetic baseline to measure the real-world accuracy gap.

```python
from sagemaker.huggingface import HuggingFace

estimator = HuggingFace(
    entry_point="train.py",
    instance_type="ml.g5.2xlarge",
    instance_count=1,
    transformers_version="4.46.1",
    pytorch_version="2.3.0",
    py_version="py311",
    hyperparameters={"model_id": "mistralai/Mistral-Nemo-Base-2407", "epochs": 1}
)
estimator.fit({"train": "s3://your-telco-llm-bucket/data/train.jsonl"})
```

---

### 8. Deploy and Run the Ensemble

**AWS Services: Amazon SageMaker Endpoints, Amazon EC2, AWS Outposts**

Based on the scenario matrix, deploy Ministral 3 14B and Gemma 3 12B as complementary experts:

- Ministral 3 14B → core network and transport scenarios
- Gemma 3 12B → RAN/radio scenarios

**Option A — SageMaker Real-Time Endpoint (managed, scalable):**

```python
from sagemaker.huggingface import HuggingFaceModel

model = HuggingFaceModel(model_data="s3://your-telco-llm-bucket/adapters/ministral/",
                          role="arn:aws:iam::ACCOUNT:role/SageMakerRole",
                          transformers_version="4.46.1", pytorch_version="2.3.0")
predictor = model.deploy(instance_type="ml.g5.2xlarge", initial_instance_count=1)
result = predictor.predict({"inputs": log_text})
```

**Option B — EC2 self-hosted (lowest cost, edge-friendly):**

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

base = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-Nemo-Base-2407",
                                             torch_dtype="bfloat16", device_map="auto")
model = PeftModel.from_pretrained(base, "./ministral-3-14b-lora-adapter")

def infer(log_text):
    inputs = tokenizer(log_text, return_tensors="pt").to("cuda")
    return model.generate(**inputs, max_new_tokens=256)
```

**Option C — AWS Outposts / AWS Local Zones (for on-premise telco edge):**
Deploy the EC2 instance on an [AWS Outpost](https://aws.amazon.com/outposts/) rack inside the operator's data center. The model runs on-premise with no data leaving the facility, meeting data residency requirements.

---

## 9. Generate Reports

**Output: HTML report and JavaScript-based PPT presentation**

After scoring, generate visual reports from `results.json` for sharing and presentation.

### HTML Report

Use `src/report_html.py` to produce a self-contained HTML file with metrics tables and per-scenario charts:

```python
import json
from pathlib import Path

def generate_html_report(results_path="results/results.json", output_path="reports/report.html"):
    results = json.loads(Path(results_path).read_text())

    rows = "\n".join(
        f"<tr><td>{r['model']}</td><td>{r['f1']:.3f}</td>"
        f"<td>{r['precision']:.3f}</td><td>{r['recall']:.3f}</td>"
        f"<td>{r['exact_match']:.3f}</td></tr>"
        for r in results
    )

    html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>Benchmark Results</title>
<style>body{{font-family:sans-serif;padding:2rem}}
table{{border-collapse:collapse;width:100%}}
th,td{{border:1px solid #ccc;padding:8px;text-align:left}}
th{{background:#f4f4f4}}</style></head>
<body><h1>3GPP RCA Benchmark Results</h1>
<table><thead><tr><th>Model</th><th>F1</th><th>Precision</th><th>Recall</th><th>Exact Match</th></tr></thead>
<tbody>{rows}</tbody></table></body></html>"""

    Path(output_path).parent.mkdir(exist_ok=True)
    Path(output_path).write_text(html)
    print(f"HTML report saved to {output_path}")
```

### JavaScript PPT Presentation

Use `src/report_ppt.js` to generate a native `.pptx` file using [pptxgenjs](https://gitbrent.github.io/PptxGenJS/):

```bash
npm install pptxgenjs
```

```javascript
// src/report_ppt.js
import pptxgen from "pptxgenjs";
import { readFileSync } from "fs";

const results = JSON.parse(readFileSync("results/results.json", "utf8"));
const prs = new pptxgen();

// Title slide
const title = prs.addSlide();
title.addText("3GPP RCA Benchmark", { x: 1, y: 1.5, fontSize: 36, bold: true });
title.addText("Fine-tuned SLMs vs Frontier Models", { x: 1, y: 2.5, fontSize: 20 });

// One slide per model
for (const r of results) {
  const slide = prs.addSlide();
  slide.addText(r.model, { x: 0.5, y: 0.3, fontSize: 28, bold: true });
  slide.addTable(
    [
      [{ text: "Metric", options: { bold: true } }, { text: "Score", options: { bold: true } }],
      ["F1",           r.f1.toFixed(3)],
      ["Precision",    r.precision.toFixed(3)],
      ["Recall",       r.recall.toFixed(3)],
      ["Exact Match",  r.exact_match.toFixed(3)],
    ],
    { x: 0.5, y: 1.2, w: 6, colW: [3, 3] }
  );
}

await prs.writeFile({ fileName: "reports/presentation.pptx" });
console.log("PPT saved to reports/presentation.pptx");
```

Run it:

```bash
node src/report_ppt.js
# Output: reports/presentation.pptx
```

Run both after scoring:

```bash
python src/report_html.py
node src/report_ppt.js
# Outputs: reports/report.html, reports/presentation.pptx
```

Upload to S3 for sharing:

```bash
aws s3 cp reports/ s3://your-telco-llm-bucket/reports/ --recursive
```

---

## Glossary: Concepts & Acronyms

### Models & Architecture

**LLM (Large Language Model)**
A neural network trained on massive amounts of text data to understand and generate human language. Examples: Claude, GPT-4. These models have billions of parameters and require significant compute to run.

**SLM (Small Language Model)**
A smaller, more efficient version of an LLM — typically 1B–14B parameters. Easier to deploy on-premise or at the edge, and much cheaper to run per inference. The trade-off is they may need fine-tuning to match frontier model quality on specific tasks.

**14B**
14 billion parameters. A parameter is a numerical weight inside the neural network that gets adjusted during training. More parameters generally means more capability, but also more memory and compute required.

**Foundation Model**
A large pre-trained model (like Claude or Nova Pro) that can be used as-is for many tasks without any additional training. Also called a frontier model when referring to the most capable, state-of-the-art versions.

**Frontier Model**
The most capable, state-of-the-art foundation models available — e.g., Claude 4.6 Opus. They perform well across a wide range of tasks out of the box but are expensive to run at scale.

---

### Fine-Tuning

**Fine-Tuning**
The process of taking a pre-trained model and continuing to train it on a smaller, task-specific dataset. This teaches the model to specialize in a particular domain (e.g., 3GPP log analysis) without training from scratch.

**LoRA (Low-Rank Adaptation)**
A parameter-efficient fine-tuning technique. Instead of updating all billions of weights in the model, LoRA adds small trainable matrices to specific layers. This dramatically reduces memory usage and training time while achieving results close to full fine-tuning.

**QLoRA (Quantized LoRA)**
An extension of LoRA that also quantizes (compresses) the base model weights to 4-bit precision before fine-tuning. This allows fine-tuning large models on less GPU memory — e.g., Qwen3-14B on 4× L4 GPUs instead of requiring A100s.

**BF16 (Brain Float 16)**
A 16-bit floating point number format used during training to reduce memory usage while maintaining numerical stability. An alternative to the standard FP32 (32-bit) format.

**Training Steps**
One step = one batch of training examples processed and used to update the model weights. 325 steps with 1,300 examples means the model saw the full dataset roughly once (1 epoch).

**Synthetic Data**
Training data that is artificially generated rather than collected from real-world systems. In this benchmark, the 3GPP logs and failure scenarios were generated programmatically to be realistic but not from a live network.

---

### Prompting Strategies

**Zero-Shot**
Asking the model to perform a task with no examples provided — just the instruction and the input. Tests the model's raw pre-trained knowledge.

**Few-Shot**
Providing a small number of input/output examples (here: 5) in the prompt before asking the model to handle a new input. Helps the model understand the expected format and reasoning pattern.

**CoT (Chain-of-Thought)**
A prompting technique where the model is encouraged to reason step-by-step before giving a final answer. Improves accuracy on complex reasoning tasks. Combined with few-shot: "5-shot + CoT".

---

### Evaluation Metrics

**F1 Score**
The harmonic mean of Precision and Recall. A score of 1.0 is perfect; 0.0 is worst. It balances both false positives and false negatives, making it a reliable single metric for classification tasks.

**Precision (P)**
Of all the error codes the model predicted, what fraction were correct? High precision = few false alarms.

**Recall (R)**
Of all the actual error codes in the ground truth, what fraction did the model find? High recall = few missed errors.

**Exact Match (EM)**
The percentage of test cases where the model's output exactly matched the expected JSON output — no partial credit. A stricter metric than F1.

**PERFECT (in Scenario Matrix)**
The model achieved 100% F1 on every test case within that failure scenario.

**ALL FP (All False Positives)**
The model flagged errors that weren't there — it predicted root causes that don't exist in the ground truth. The model is over-triggering.

**NEAR FAIL**
Borderline performance — the model is close to failing the scenario, with significantly degraded F1.

---

### Telco & 3GPP Concepts

**3GPP (3rd Generation Partnership Project)**
The international standards body that defines the technical specifications for mobile networks, including 4G LTE and 5G. 3GPP logs are the protocol-level messages exchanged between network components.

**5G SA (5G Standalone)**
A 5G network architecture that operates independently of 4G infrastructure, using a fully cloud-native 5G core. "SA" contrasts with "NSA" (Non-Standalone), which relies on a 4G core.

**NAS (Non-Access Stratum)**
A protocol layer in 5G/4G that handles signaling between the mobile device (UE) and the core network — covering authentication, session management, and mobility.

**NGAP (Next Generation Application Protocol)**
The interface protocol between the 5G radio access network (gNB) and the 5G core (AMF). Carries control plane messages for handovers, paging, and UE context management.

**RRC (Radio Resource Control)**
A protocol between the UE (device) and the base station (gNB) that manages radio connections — including setup, reconfiguration, and release of radio bearers.

**UPF (User Plane Function)**
The component in the 5G core network responsible for routing and forwarding user data packets. A UPF degradation cascade means a failure in the UPF that triggers a chain of downstream failures.

**RCA (Root Cause Analysis)**
The process of identifying the original source of a failure in a system, as opposed to the symptoms or secondary effects it causes.

**NOC (Network Operations Center)**
The team responsible for monitoring, managing, and maintaining a telecom operator's network. They triage alarms and perform root cause analysis on network incidents.

**Sympathetic Noise**
In the context of network alarms, these are secondary failures or alerts that are triggered as a consequence of a root cause failure — not the cause itself. Examples: heartbeat timeouts, keepalive failures, cascading errors. Filtering these out is critical to accurate RCA.

---

### Infrastructure & Cost

**L40S / L4**
NVIDIA GPU models. The L40S is a high-memory GPU suited for BF16 training of large models. The L4 is a smaller, more cost-efficient GPU suited for QLoRA (4-bit) workloads.

**g6e.2xlarge**
An AWS EC2 instance type equipped with a single L40S GPU. Used here for the full training pipeline at $1.86/hr.

**LoRA Adapter**
The small set of trained weights produced by LoRA fine-tuning. Instead of saving a full copy of the model, you save only the adapter (a few hundred MB) and load it on top of the base model at inference time.

**Inference Cost per 1,000**
The dollar cost to run the model on 1,000 input/output requests. For API-based frontier models this is a per-token charge. For self-hosted fine-tuned SLMs, it's the compute cost of running the GPU instance.

**Amazon Bedrock**
AWS's managed service for accessing foundation models (Claude, Nova, Titan, etc.) via API — no infrastructure to manage. Pay per token used.

**SageMaker / EC2**
AWS services for running custom ML workloads. SageMaker provides managed training and inference infrastructure; EC2 gives raw virtual machine access. Both allow self-hosting fine-tuned models with no per-token cost.

**Edge Deployment**
Running a model close to where data is generated — e.g., inside a telco's data center or on an AWS Outpost — rather than sending data to a central cloud API. Reduces latency, cost, and data sovereignty concerns.

**CUDA (Compute Unified Device Architecture)**
NVIDIA's parallel computing platform and programming model that lets software use the GPU for general-purpose computation — not just graphics.

In the context of this benchmark:
- PyTorch uses CUDA to run model training on the GPU. When you see `torch.cuda.is_available()`, it's checking whether a CUDA-capable GPU is present and accessible.
- The `cu121` / `cu128` suffixes in DLC image names (e.g. `pytorch2.3.0-cu121`) refer to the CUDA toolkit version bundled in that container — `cu121` = CUDA 12.1, `cu128` = CUDA 12.8.
- The CUBLAS regression mentioned in this guide (`torch 2.10+cu128`) is a bug in CUDA 12.8's linear algebra library (cuBLAS) that corrupts BF16/FP16 matrix multiplications during training — which is why we pin to `pytorch2.3.0+cu121`.
- `device_map={"": 0}` in `train.py` means "put everything on `cuda:0`" — the first (and only) GPU on the `ml.g5.2xlarge` instance.

**Deep Learning AMI (Amazon Machine Image)**
A pre-configured EC2 virtual machine image provided by AWS that comes with CUDA, PyTorch, TensorFlow, and common ML libraries pre-installed. Saves hours of environment setup when launching a GPU instance.

**EBS (Elastic Block Store)**
AWS's persistent block storage for EC2 instances — essentially a virtual hard drive. Used here to store model weights, datasets, and training checkpoints. `gp3` is the recommended general-purpose SSD volume type.

**IAM Role (Identity and Access Management Role)**
An AWS identity with specific permissions attached to it. Assigning an IAM role to an EC2 instance allows the code running on that instance to call other AWS services (like S3 or Bedrock) securely, without hardcoding credentials.

**IAM Policy**
A document that defines what actions an IAM role or user is allowed to perform on which AWS resources. For example, `AmazonS3FullAccess` allows reading and writing to any S3 bucket; `AmazonBedrockFullAccess` allows calling any Bedrock model.

**boto3**
The official AWS SDK for Python. Used to interact with AWS services programmatically — calling Bedrock APIs, reading/writing S3 files, launching EC2 instances, etc.

**Hugging Face**
An open-source platform and company that hosts thousands of pre-trained ML models (including Ministral, Qwen3, Gemma) and provides the `transformers`, `peft`, and `trl` Python libraries used for fine-tuning.

**TRL (Transformer Reinforcement Learning)**
A Hugging Face library that provides the `SFTTrainer` class — a high-level wrapper for supervised fine-tuning of language models. Simplifies the training loop significantly.

**PEFT (Parameter-Efficient Fine-Tuning)**
A Hugging Face library that implements LoRA, QLoRA, and other parameter-efficient fine-tuning methods. Provides `LoraConfig` and `get_peft_model()` to wrap any base model with LoRA adapters.

**`transformers` library**
The core Hugging Face Python library for loading, running, and fine-tuning pre-trained models. Provides `AutoModelForCausalLM`, `AutoTokenizer`, and `TrainingArguments`.

**`bitsandbytes`**
A Python library that enables 4-bit and 8-bit quantization of model weights, required for QLoRA fine-tuning. Reduces GPU memory usage dramatically.

**`accelerate`**
A Hugging Face library that handles distributed training across multiple GPUs or machines with minimal code changes.

**`hf auth login`**
A command-line tool to authenticate with the Hugging Face Hub. Required to download gated models (models that require accepting a license agreement before access is granted). Installed via the standalone installer at `https://hf.co/cli/install.sh`; the binary is `hf` (not `huggingface-cli`).

---

### AWS Services

**Amazon S3 (Simple Storage Service)**
AWS's object storage service. Used here to store training datasets, LoRA adapter weights, and evaluation results. Data is organized into buckets and accessed via the `aws s3` CLI or `boto3`.

**S3 Bucket Policy**
A JSON document attached to an S3 bucket that controls who can access it. Used in the cross-account data sharing scenario to grant a partner's AWS account read access to an operator's data bucket.

**AWS Resource Access Manager (RAM)**
An AWS service that allows sharing resources (like S3 buckets, VPCs, subnets) across AWS accounts without copying data. Useful for multi-account pipelines where the operator owns the data account.

**VPC (Virtual Private Cloud)**
A logically isolated network within AWS where you launch resources like EC2 instances. Controls inbound/outbound traffic via security groups and network ACLs.

**AWS PrivateLink**
A networking feature that allows services in one VPC to be accessed from another VPC (or on-premise network) without traffic traversing the public internet. Used here to ensure operator data never leaves their private network environment.

**AWS Systems Manager Session Manager**
A browser-based or CLI shell into EC2 instances that doesn't require opening port 22 (SSH). More secure than traditional SSH as it uses IAM for access control and logs all sessions.

**Amazon SageMaker Training Jobs**
A fully managed service for running ML training workloads. You provide a training script and dataset location (S3), specify an instance type, and SageMaker handles provisioning, running, monitoring, and terminating the instance automatically.

**Amazon SageMaker Real-Time Endpoint**
A managed, auto-scaling HTTPS endpoint for serving ML model predictions. You deploy a model from S3 and SageMaker handles load balancing, health checks, and scaling. Accessed via the `predictor.predict()` API.

**`HuggingFace` SageMaker Estimator**
A SageMaker SDK class that simplifies running Hugging Face training scripts as SageMaker Training Jobs. Handles container selection, instance provisioning, and S3 artifact management automatically.

**`ml.g5.2xlarge`**
A SageMaker instance type with a single NVIDIA A10G GPU (24GB VRAM). The managed SageMaker equivalent of the EC2 `g6e.2xlarge` used in this benchmark.

**AWS Outposts**
A fully managed service that extends AWS infrastructure, services, and APIs to on-premise data centers or co-location facilities. Allows running EC2 instances and other AWS services physically inside an operator's building — ideal for data residency and low-latency requirements.

**AWS Local Zones**
AWS infrastructure placed in metropolitan areas closer to end users, extending a parent AWS region. Provides low-latency access to AWS services for latency-sensitive workloads without requiring a full Outpost deployment.

**Data Residency**
A regulatory or contractual requirement that data must be stored and processed within a specific geographic boundary (country or region). Relevant for telco operators who cannot send network data to public cloud regions outside their jurisdiction.

**Cross-Account Access**
A pattern where resources in one AWS account (e.g., an operator's data account) are accessed by workloads running in a different AWS account (e.g., a research or pipeline account). Managed via IAM roles, S3 bucket policies, or AWS RAM.

**`sklearn` (scikit-learn)**
A popular Python machine learning library. Used here specifically for its `f1_score`, `precision_score`, and `recall_score` functions to compute evaluation metrics against the ground-truth test set.

**JSONL (JSON Lines)**
A file format where each line is a valid JSON object. Commonly used for ML datasets because it's easy to stream line-by-line without loading the entire file into memory. Training and test datasets in this pipeline are stored as `.jsonl` files.
