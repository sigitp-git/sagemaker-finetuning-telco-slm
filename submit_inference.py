"""
Submit a SageMaker Training Job for SLM inference (batch prediction).
Uses Training Job API (not Processing) to leverage existing GPU quota.
Loads base model + LoRA adapter, runs predictions on test set, uploads results to S3.

Usage:
  python3 submit_inference.py \
    --role arn:aws:iam::ACCOUNT_ID:role/SageMakerRole \
    --bucket your-telco-llm-bucket \
    --model_id mistralai/Mistral-Nemo-Base-2407 \
    [--instance_type ml.g5.2xlarge] \
    [--hf_token HF_TOKEN]
"""
import argparse, time, boto3, sagemaker
from sagemaker.estimator import Estimator

# Same DLC as training — has transformers, peft, bitsandbytes, torch
INFERENCE_IMAGE_URI = (
    "763104351884.dkr.ecr.{region}.amazonaws.com/"
    "huggingface-pytorch-training:2.8.0-transformers4.56.2-gpu-py312-cu129-ubuntu22.04"
)

MODEL_DEFAULTS = {
    "mistralai/Mistral-Nemo-Base-2407": {"instance_type": "ml.g5.2xlarge"},
    "Qwen/Qwen3-14B":                   {"instance_type": "ml.g5.12xlarge"},
    "google/gemma-3-12b-pt":            {"instance_type": "ml.g5.2xlarge"},
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--role",          required=True,  help="SageMaker IAM role ARN")
    parser.add_argument("--bucket",        required=True,  help="S3 bucket name")
    parser.add_argument("--region",        default="us-east-1")
    parser.add_argument("--model_id",      default="mistralai/Mistral-Nemo-Base-2407")
    parser.add_argument("--instance_type", default=None,   help="Override default instance type")
    parser.add_argument("--hf_token",      default=None,   help="HF token for gated models (Gemma)")
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

    defaults = MODEL_DEFAULTS.get(args.model_id, {"instance_type": "ml.g5.2xlarge"})
    instance_type = args.instance_type or defaults["instance_type"]

    # Derive model slug for naming
    slug = args.model_id.split("/")[-1].lower().replace(".", "-").replace("_", "-")
    output_filename = f"preds_{slug}_slm.jsonl"

    boto_session = boto3.Session(region_name=args.region)
    sm_session = sagemaker.Session(boto_session=boto_session)
    sm_client = boto_session.client("sagemaker")

    # Find the latest training job output for this model
    s3 = boto_session.client("s3")
    prefix = f"output/{slug}/"
    resp = s3.list_objects_v2(Bucket=args.bucket, Prefix=prefix, Delimiter="/")
    job_prefixes = sorted([p["Prefix"] for p in resp.get("CommonPrefixes", [])])
    if not job_prefixes:
        print(f"ERROR: No training output found at s3://{args.bucket}/{prefix}")
        return
    latest_job_prefix = job_prefixes[-1]
    adapter_s3_uri = f"s3://{args.bucket}/{latest_job_prefix}output/"
    print(f"Using adapter from: {adapter_s3_uri}")

    env = {}
    if args.hf_token:
        env["HF_TOKEN"] = args.hf_token

    estimator = Estimator(
        entry_point="inference_slm.py",
        source_dir="./src",
        instance_type=instance_type,
        instance_count=1,
        role=args.role,
        sagemaker_session=sm_session,
        image_uri=INFERENCE_IMAGE_URI.format(region=args.region),
        hyperparameters={
            "model_id": args.model_id,
            "output_filename": output_filename,
        },
        environment=env,
        output_path=f"s3://{args.bucket}/inference-output/{slug}/",
        base_job_name=f"telco-rca-infer-{slug[:20]}",
        max_run=7200,  # 2 hour max
    )

    estimator.fit(
        {
            "test": f"s3://{args.bucket}/data/test.jsonl",
            "adapter": adapter_s3_uri,
        },
        wait=False,
    )

    job_name = estimator.latest_training_job.name
    console_url = f"https://console.aws.amazon.com/sagemaker/home?region={args.region}#/jobs/{job_name}"
    print(f"\nJob submitted : {job_name}")
    print(f"Instance      : {instance_type}")
    print(f"Model         : {args.model_id}")
    print(f"Adapter       : {adapter_s3_uri}")
    print(f"Output        : s3://{args.bucket}/inference-output/{slug}/")
    print(f"Console       : {console_url}")
    print(f"\nPoll status   : aws sagemaker describe-training-job --training-job-name {job_name} --query TrainingJobStatus --output text")

    if args.wait:
        final = poll_job(sm_client, job_name)
        print(f"\nFinal status: {final}")


if __name__ == "__main__":
    main()
