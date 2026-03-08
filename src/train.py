"""
SageMaker entry point for LoRA/QLoRA fine-tuning of 3GPP RCA models.
Supports: Mistral-Nemo-Base-2407 (QLoRA 4-bit), Qwen3-14B (QLoRA 4-bit), Gemma-3-12B (BF16 LoRA)

Note: Mistral-Nemo BF16 ≈ 24GB which exactly fills the A10G — no headroom for training.
      Use 4-bit QLoRA for Mistral-Nemo on ml.g5.2xlarge.

Usage (local):
  python src/train.py --model_id mistralai/Mistral-Nemo-Base-2407 --use_4bit True

Usage (SageMaker): entry_point="train.py", source_dir="./src"
  hyperparameters passed as CLI args by SageMaker.
"""

import argparse
import os
import json
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer, SFTConfig, DataCollatorForCompletionOnlyLM

LORA_CONFIGS = {
    # model_id -> (r, lora_alpha, target_modules); Mistral-Nemo uses GQA — target all 4 proj layers
    "mistralai/Mistral-Nemo-Base-2407": (16, 32, ["q_proj", "k_proj", "v_proj", "o_proj"]),
    "Qwen/Qwen3-14B":                   (16, 32, ["q_proj", "v_proj", "k_proj", "o_proj"]),
    "google/gemma-3-12b-pt":            (16, 32, ["q_proj", "v_proj", "k_proj", "o_proj"]),
    "google/gemma-3-12b-it":            (16, 32, ["q_proj", "v_proj", "k_proj", "o_proj"]),
}

# Response template — SFTTrainer masks loss on all tokens before this string,
# so the model only learns to generate the completion (JSON array) after it.
# This is the key fix (Option C) for Qwen3 and Gemma which failed to produce
# structured output when trained on the full prompt+completion as one text field.
RESPONSE_TEMPLATE = "### Root Cause\n"


def format_example(example):
    """Convert JSONL example to instruction-following prompt."""
    log = example["log"]
    label = json.dumps(example["root_cause"])
    return {
        "text": (
            "### Instruction\nAnalyze the following 3GPP signaling log and identify the root cause.\n\n"
            f"### Log\n{log}\n\n"
            f"### Root Cause\n{label}"
        )
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", default="mistralai/Mistral-Nemo-Base-2407")
    parser.add_argument("--max_steps", type=int, default=325)
    parser.add_argument("--bf16", type=lambda x: x.lower() == "true", default=True)
    parser.add_argument("--per_device_train_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--use_4bit", type=lambda x: x.lower() == "true", default=False)
    parser.add_argument("--output_dir", default=os.environ.get("SM_OUTPUT_DATA_DIR", "./output"))
    parser.add_argument("--train_data", default=os.environ.get("SM_CHANNEL_TRAIN", "./data"))
    args = parser.parse_args()

    train_file = os.path.join(args.train_data, "train.jsonl") \
        if os.path.isdir(args.train_data) else args.train_data

    print(f"Model: {args.model_id}")
    print(f"Train file: {train_file}")
    print(f"Max steps: {args.max_steps} | BF16: {args.bf16} | 4-bit: {args.use_4bit}")

    # Authenticate with Hugging Face for gated models (e.g. Gemma)
    hf_token = os.environ.get("HF_TOKEN")
    if hf_token:
        from huggingface_hub import login
        login(token=hf_token)
        print("Authenticated with Hugging Face (gated model access)")

    # Quantization config for QLoRA
    bnb_config = None
    if args.use_4bit:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )

    # For BF16 LoRA: load entirely on CPU first to avoid OOM during weight loading.
    # Mistral-Nemo 12B BF16 ≈ 24GB — exactly the A10G limit, leaving zero headroom.
    # Strategy: load on CPU → wrap with PEFT on CPU → move to GPU → enable gradient checkpointing.
    # For QLoRA (4-bit): device_map="auto" is required by bitsandbytes.
    load_kwargs = dict(
        torch_dtype=torch.bfloat16 if args.bf16 and not args.use_4bit else "auto",
        quantization_config=bnb_config,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )
    if args.use_4bit:
        load_kwargs["device_map"] = "auto"
    else:
        load_kwargs["device_map"] = "cpu"  # load fully on CPU, move to GPU after PEFT wrapping

    model = AutoModelForCausalLM.from_pretrained(args.model_id, **load_kwargs)
    tokenizer = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    r, alpha, targets = LORA_CONFIGS.get(args.model_id, (16, 32, ["q_proj", "v_proj"]))
    lora_config = LoraConfig(
        r=r, lora_alpha=alpha, target_modules=targets,
        lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # QLoRA 4-bit: embedding outputs don't carry gradients by default — enable them so the
    # backward pass can flow through the frozen base model into the LoRA adapters.
    # Without this, Qwen3 (and other models on newer transformers) hits:
    #   RuntimeError: element 0 of tensors does not require grad and does not have a grad_fn
    if args.use_4bit:
        model.enable_input_require_grads()

    # Move to GPU after PEFT wrapping (BF16 only — QLoRA already placed by device_map)
    if not args.use_4bit:
        model = model.to("cuda")

    # Gradient checkpointing trades compute for memory — essential for BF16 on 24GB VRAM,
    # and also helps QLoRA on multi-GPU setups. Enable for all configurations.
    model.gradient_checkpointing_enable()

    dataset = load_dataset("json", data_files=train_file, split="train")
    dataset = dataset.map(format_example, remove_columns=dataset.column_names)

    training_args = SFTConfig(
        output_dir=args.output_dir,
        max_steps=args.max_steps,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        bf16=args.bf16 and not args.use_4bit,
        fp16=False,
        gradient_checkpointing=True,
        logging_steps=25,
        save_steps=args.max_steps,
        warmup_ratio=0.05,
        lr_scheduler_type="cosine",
        report_to="none",
        max_length=1024,
        dataset_text_field="text",
    )

    # Option C: Use DataCollatorForCompletionOnlyLM so the model only trains on
    # the completion tokens (the JSON array after "### Root Cause\n"), not the prompt.
    # This teaches the model precisely: "when you see ### Root Cause\n, output a JSON array."
    collator = DataCollatorForCompletionOnlyLM(
        response_template=RESPONSE_TEMPLATE,
        tokenizer=tokenizer,
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        processing_class=tokenizer,
        data_collator=collator,
    )
    trainer.train()

    adapter_dir = os.path.join(args.output_dir, "adapter")
    model.save_pretrained(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)
    print(f"Adapter saved to {adapter_dir}")


if __name__ == "__main__":
    main()
