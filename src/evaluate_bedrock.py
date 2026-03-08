"""
Frontier model evaluation via Amazon Bedrock.
Runs Claude Opus 4.6 and Nova Pro against the test set with three prompt strategies:
zero-shot, 5-shot, and 5-shot + CoT.

Uses the Bedrock Converse API for uniform request/response format across models.

Usage:
  python src/evaluate_bedrock.py --model nova --strategy zero_shot
  python src/evaluate_bedrock.py --model claude --strategy five_shot_cot
"""

import argparse
import json
import os
import sys
import time
import boto3

sys.path.insert(0, os.path.dirname(__file__))
from filter import extract_root_cause_from_text

REGION = "us-east-1"

MODEL_IDS = {
    "claude": "us.anthropic.claude-opus-4-6-v1",
    "nova":   "amazon.nova-pro-v1:0",
}

SYSTEM_PROMPT = (
    "You are a 5G core network expert. Analyze 3GPP signaling logs and identify "
    "the root cause. Respond with ONLY a JSON array containing one label from: "
    "[core_network_failure, authentication_failure, normal, handover_failure, "
    "congestion, qos_violation, transport_jitter, radio_failure]"
)


def load_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def build_messages(log_text: str, few_shot: list, use_cot: bool) -> list:
    messages = []
    for ex in few_shot:
        messages.append({"role": "user",      "content": [{"text": ex["log"]}]})
        messages.append({"role": "assistant", "content": [{"text": json.dumps(ex["root_cause"])}]})
    suffix = "\n\nThink step by step, then output the JSON array." if use_cot else ""
    messages.append({"role": "user", "content": [{"text": log_text + suffix}]})
    return messages


def invoke_model(bedrock, model_id: str, messages: list) -> str:
    """Call Bedrock Converse API — works uniformly for Claude and Nova."""
    resp = bedrock.converse(
        modelId=model_id,
        system=[{"text": SYSTEM_PROMPT}],
        messages=messages,
        inferenceConfig={"maxTokens": 512, "temperature": 0.0},
    )
    return resp["output"]["message"]["content"][0]["text"]


def evaluate(test_data: list, model_key: str, strategy: str, out_path: str):
    bedrock = boto3.client("bedrock-runtime", region_name=REGION)
    model_id = MODEL_IDS[model_key]
    use_cot = "cot" in strategy
    n_shot = 5 if "five" in strategy else 0
    few_shot_pool = test_data[:n_shot]
    eval_data = test_data[n_shot:]

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    written = 0
    with open(out_path, "w") as f:
        for i, ex in enumerate(eval_data):
            messages = build_messages(ex["log"], few_shot_pool if n_shot > 0 else [], use_cot)
            try:
                text = invoke_model(bedrock, model_id, messages)
                predicted = extract_root_cause_from_text(text)
            except Exception as e:
                print(f"  [WARN] example {i}: {e}")
                predicted = ["normal"]
                text = ""

            f.write(json.dumps({"root_cause": predicted, "raw": text if use_cot else ""}) + "\n")
            written += 1
            if (i + 1) % 50 == 0:
                print(f"  {i+1}/{len(eval_data)} done")
            time.sleep(0.5)

    print(f"Saved {written} predictions to {out_path}")
    return out_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["claude", "nova"], default="nova")
    parser.add_argument("--strategy", choices=["zero_shot", "five_shot", "five_shot_cot"], default="zero_shot")
    parser.add_argument("--test", default="data/test.jsonl")
    parser.add_argument("--out_dir", default="results")
    args = parser.parse_args()

    test_data = load_jsonl(args.test)
    out_path = os.path.join(args.out_dir, f"preds_{args.model}_{args.strategy}.jsonl")
    print(f"Evaluating {args.model} ({MODEL_IDS[args.model]}) / {args.strategy} on {len(test_data)} examples...")
    evaluate(test_data, args.model, args.strategy, out_path)
    print(f"Next: python src/evaluate.py --predictions {out_path} --model {args.model} --strategy {args.strategy}")


if __name__ == "__main__":
    main()
