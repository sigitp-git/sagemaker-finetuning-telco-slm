"""
Submit a SageMaker Training Job for LoRA/QLoRA fine-tuning.

Usage:
  python submit_training.py \
    --role arn:aws:iam::ACCOUNT_ID:role/SageMakerRole \
    --bucket your-telco-llm-bucket \
    [--model_id mistralai/Mistral-Nemo-Base-2407] \
    [--instance_type ml.g5.2xlarge] \
    [--max_steps 325] \
    [--use_4bit]        # enable QLoRA for Qwen3-14B
"""
import argparse
import time
import boto3
import sagemaker
from sagemaker.huggingface import HuggingFace

MODEL_DEFAULTS = {
    "mistralai/Mistral-Nemo-Base-2407": {"instance_type": "ml.g5.2xlarge",  "use_4bit": False},
    "Qwen/Qwen3-14B":                   {"instance_type": "ml.g5.12xlarge", "use_4bit": True},
    "google/gemma-3-12b-pt":            {"instance_type": "ml.g5.2xlarge",  "use_4bit": False},
}

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--role",          required=True,  help="SageMaker IAM role ARN")
    parser.add_argument("--bucket",        required=True,  help="S3 bucket name")
    parser.add_argument("--region",        default="us-east-1")
    parser.add_argument("--model_id",      default="mistralai/Mistral-Nemo-Base-2407")
    parser.add_argument("--instance_type", default=None,   help="Override default instance type")
    parser.add_argument("--max_steps",     type=int, default=325)
    parser.add_argument("--use_4bit",      action="store_true", help="Enable QLoRA 4-bit (for Qwen3-14B)")
    parser.add_argument("--wait",          action="store_true", help="Block until job completes")
    return parser.parse_args()


def poll_job(sm_client, job_name, interval=60):
    """Poll training job status until terminal state."""
    print(f"\nPolling job: {job_name}")
    while True:
        resp = sm_client.describe_training_job(TrainingJobName=job_name)
        status = resp["TrainingJobStatus"]
        elapsed = resp.get("TrainingTimeInSeconds", 0)
        print(f"  [{time.strftime('%H:%M:%S')}] Status: {status}  |  Elapsed: {elapsed}s")
        if status in ("Completed", "Failed", "Stopped"):
            if status == "Failed":
                print(f"  Failure reason: {resp.get('FailureReason', 'N/A')}")
            return status
        time.sleep(interval)


def main():
    args = parse_args()

    defaults = MODEL_DEFAULTS.get(args.model_id, {"instance_type": "ml.g5.2xlarge", "use_4bit": False})
    instance_type = args.instance_type or defaults["instance_type"]
    use_4bit = args.use_4bit or defaults["use_4bit"]

    # Derive a short model slug for naming (e.g. "mistral-nemo", "qwen3-14b")
    slug = args.model_id.split("/")[-1].lower().replace(".", "-").replace("_", "-")
    base_job_name = f"telco-rca-{slug[:28]}"

    boto_session = boto3.Session(region_name=args.region)
    sm_session = sagemaker.Session(boto_session=boto_session)
    sm_client = boto_session.client("sagemaker")

    estimator = HuggingFace(
        entry_point="train.py",
        source_dir="./src",
        instance_type=instance_type,
        instance_count=1,
        role=args.role,
        sagemaker_session=sm_session,
        transformers_version="4.46.1",
        pytorch_version="2.3.0",
        py_version="py311",
        hyperparameters={
            "model_id":   args.model_id,
            "max_steps":  args.max_steps,
            "bf16":       not use_4bit,
            "use_4bit":   use_4bit,
        },
        output_path=f"s3://{args.bucket}/output/{slug}/",
        base_job_name=base_job_name,
    )

    estimator.fit(
        {"train": f"s3://{args.bucket}/data/train.jsonl"},
        wait=False,
    )

    job_name = estimator.latest_training_job.name
    console_url = f"https://console.aws.amazon.com/sagemaker/home?region={args.region}#/jobs/{job_name}"
    print(f"\nJob submitted : {job_name}")
    print(f"Instance      : {instance_type}")
    print(f"Model         : {args.model_id}")
    print(f"Output        : s3://{args.bucket}/output/{slug}/")
    print(f"Console       : {console_url}")
    print(f"\nPoll status   : aws sagemaker describe-training-job --training-job-name {job_name} --query TrainingJobStatus --output text")

    if args.wait:
        final = poll_job(sm_client, job_name)
        print(f"\nFinal status: {final}")


if __name__ == "__main__":
    main()
