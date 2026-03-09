"""
SageMaker Processing Job entry point for SLM inference.
Loads base model + LoRA adapter in QLoRA 4-bit, runs predictions on test set.

Usage (SageMaker Processing):
  Invoked by submit_inference.py — not run directly.

Usage (local):
  python3 src/inference_slm.py \
    --model_id mistralai/Mistral-Nemo-Base-2407 \
    --adapter_dir ./adapters/mistral-nemo-base-2407 \
    --test_file data/test.jsonl \
    --output_file results/preds_mistral-nemo-base-2407_slm.jsonl
"""
import argparse, json, os, sys, torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

PROMPT_TEMPLATE = (
    "### Instruction\n"
    "Analyze the following 3GPP signaling log and identify the root cause.\n\n"
    "### Log\n{log}\n\n"
    "### Root Cause\n"
)

# Qwen3 uses its native chat template for much better structured output
CHAT_TEMPLATE_MODELS = {"Qwen/Qwen3-14B"}

SYSTEM_PROMPT = (
    "You are a 3GPP root cause analysis assistant. "
    "Given a signaling log, respond with ONLY a JSON array of root cause labels. "
    "Valid labels: core_network_failure, authentication_failure, normal, "
    "handover_failure, congestion, qos_violation, transport_jitter, radio_failure. "
    "Example: [\"congestion\"]"
)

QWEN3_PROMPT_TEMPLATE = (
    "<|im_start|>system\n" + SYSTEM_PROMPT + "\n/no_think<|im_end|>\n"
    "<|im_start|>user\n{log}<|im_end|>\n"
    "<|im_start|>assistant\n"
)


def load_jsonl(path):
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]


def parse_root_cause(text):
    """Extract root cause from generated text after '### Root Cause' marker."""
    # Add filter.py's directory to path for import — works both locally (src/)
    # and on SageMaker Processing (/opt/ml/processing/input/code/)
    sys.path.insert(0, os.path.dirname(__file__))
    sys.path.insert(0, "/opt/ml/processing/input/code")
    from filter import extract_root_cause_from_text
    return extract_root_cause_from_text(text)


def extract_adapter(adapter_input_dir):
    """Find and extract output.tar.gz from SageMaker training output, return adapter path."""
    import tarfile, glob
    # Look for output.tar.gz anywhere under the input directory
    tarballs = glob.glob(os.path.join(adapter_input_dir, "**", "output.tar.gz"), recursive=True)
    if not tarballs:
        # Maybe the adapter files are already extracted (local run)
        if os.path.exists(os.path.join(adapter_input_dir, "adapter_config.json")):
            return adapter_input_dir
        if os.path.exists(os.path.join(adapter_input_dir, "adapter", "adapter_config.json")):
            return os.path.join(adapter_input_dir, "adapter")
        raise FileNotFoundError(f"No output.tar.gz or adapter_config.json found in {adapter_input_dir}")

    tarball = tarballs[0]
    extract_dir = os.path.join(adapter_input_dir, "_extracted")
    print(f"Extracting {tarball} to {extract_dir}")
    os.makedirs(extract_dir, exist_ok=True)
    with tarfile.open(tarball, "r:gz") as tf:
        tf.extractall(extract_dir)

    # The tarball contains adapter/ directory with the LoRA weights
    adapter_path = os.path.join(extract_dir, "adapter")
    if os.path.exists(adapter_path):
        return adapter_path
    # Fallback: check for adapter_config.json directly in extract dir
    if os.path.exists(os.path.join(extract_dir, "adapter_config.json")):
        return extract_dir
    raise FileNotFoundError(f"No adapter/ directory found in extracted tarball: {os.listdir(extract_dir)}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", required=True)
    parser.add_argument("--adapter_dir", default=os.environ.get("SM_CHANNEL_ADAPTER", "./adapter"))
    parser.add_argument("--test_file", default=None)
    parser.add_argument("--output_file", default=None)
    parser.add_argument("--output_filename", default="predictions.jsonl")
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=4)
    args = parser.parse_args()

    # Resolve paths for SageMaker Training Job environment
    if args.test_file is None:
        test_channel = os.environ.get("SM_CHANNEL_TEST", "./data")
        args.test_file = os.path.join(test_channel, "test.jsonl") \
            if os.path.isdir(test_channel) else test_channel
    if args.output_file is None:
        output_dir = os.environ.get("SM_OUTPUT_DATA_DIR", "./output")
        os.makedirs(output_dir, exist_ok=True)
        args.output_file = os.path.join(output_dir, args.output_filename)

    # Authenticate with Hugging Face for gated models
    hf_token = os.environ.get("HF_TOKEN")
    if hf_token:
        from huggingface_hub import login
        login(token=hf_token)
        print("Authenticated with Hugging Face (gated model access)")

    print(f"Model: {args.model_id}")
    print(f"Adapter input: {args.adapter_dir}")
    print(f"Test file: {args.test_file}")

    # Extract adapter from tarball if needed
    adapter_dir = extract_adapter(args.adapter_dir)
    print(f"Adapter resolved to: {adapter_dir}")
    print(f"Adapter contents: {os.listdir(adapter_dir)}")

    # Load base model in QLoRA 4-bit
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"  # left-pad for batch generation

    # Merge LoRA adapter
    print(f"Loading adapter from {adapter_dir}")
    model = PeftModel.from_pretrained(model, adapter_dir)
    model.eval()

    # Load test data
    test_data = load_jsonl(args.test_file)
    print(f"Loaded {len(test_data)} test examples")

    # Run inference in batches
    use_chat_template = args.model_id in CHAT_TEMPLATE_MODELS
    if use_chat_template:
        print(f"Using Qwen3 chat template for {args.model_id}")
        # Disable Qwen3's thinking mode — it generates <think>...</think> blocks
        # that eat up the token budget and contain all label keywords (false positives).
        # With thinking disabled, the model outputs the JSON array directly.
        if hasattr(model, 'generation_config'):
            model.generation_config.do_sample = False
        # Build stop token IDs for <|im_end|> so the model stops after the JSON answer
        stop_token_ids = tokenizer.encode("<|im_end|>", add_special_tokens=False)
        from transformers import StoppingCriteria, StoppingCriteriaList

        class StopOnImEnd(StoppingCriteria):
            def __call__(self, input_ids, scores, **kwargs):
                # Check if the last generated tokens match <|im_end|>
                if input_ids.shape[1] >= len(stop_token_ids):
                    for row in input_ids:
                        if row[-len(stop_token_ids):].tolist() == stop_token_ids:
                            return True
                return False

        stopping_criteria = StoppingCriteriaList([StopOnImEnd()])
    else:
        stopping_criteria = None

    results = []
    for i in range(0, len(test_data), args.batch_size):
        batch = test_data[i : i + args.batch_size]
        if use_chat_template:
            prompts = [QWEN3_PROMPT_TEMPLATE.format(log=ex["log"]) for ex in batch]
        else:
            prompts = [PROMPT_TEMPLATE.format(log=ex["log"]) for ex in batch]

        inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=960)
        # Remove token_type_ids — not all models accept them (e.g. Mistral)
        inputs.pop("token_type_ids", None)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        generate_kwargs = dict(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
            temperature=1.0,
            pad_token_id=tokenizer.pad_token_id,
        )
        if stopping_criteria:
            generate_kwargs["stopping_criteria"] = stopping_criteria

        with torch.no_grad():
            outputs = model.generate(**generate_kwargs)

        for j, output in enumerate(outputs):
            # Decode only the generated tokens (skip the prompt)
            prompt_len = inputs["input_ids"].shape[1]
            generated = tokenizer.decode(output[prompt_len:], skip_special_tokens=True).strip()
            root_cause = parse_root_cause(generated)
            results.append({"root_cause": root_cause, "output": generated})

        if (i // args.batch_size + 1) % 10 == 0 or i + args.batch_size >= len(test_data):
            print(f"  {min(i + args.batch_size, len(test_data))}/{len(test_data)} done")

    # Save predictions
    os.makedirs(os.path.dirname(args.output_file) or ".", exist_ok=True)
    with open(args.output_file, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")
    print(f"Saved {len(results)} predictions to {args.output_file}")


if __name__ == "__main__":
    main()
