# Fixes

## [2026-03-07] Mistral-Nemo LoRA shape mismatch

**Job:** `telco-rca-ministral-3-14b-2026-03-07-01-37-02-420`  
**Status:** Failed after ~7 min (exit code 1)

### Error

```
ValueError: Trying to set a tensor of shape torch.Size([1024, 5120]) in "weight"
(which has shape torch.Size([1280, 5120])), this looks incorrect.
```

### Root Cause

`Mistral-Nemo-Base-2407` uses Grouped Query Attention (GQA) where `v_proj` has output
dimension `num_kv_heads × head_dim = 10 × 128 = 1280`, not the full `num_heads × head_dim`
you'd expect from standard MHA. The original LoRA config only targeted `["q_proj", "v_proj"]`,
which caused PEFT to misalign adapter shapes when initializing against the GQA projection layers.

### Fix

Added `k_proj` and `o_proj` to the target modules for Mistral-Nemo in `LORA_CONFIGS` (`train.py`):

```python
# Before
"mistralai/Mistral-Nemo-Base-2407": (16, 32, ["q_proj", "v_proj"]),

# After
"mistralai/Mistral-Nemo-Base-2407": (16, 32, ["q_proj", "k_proj", "v_proj", "o_proj"]),
```

Targeting all four projection layers is the standard safe approach for GQA models — it lets
PEFT correctly infer shapes across all attention projections regardless of KV head count.


---

## [2026-03-07] SageMaker source tarball caching — fix not picked up on resubmit

**Job:** `telco-rca-mistral-nemo-base-2407-2026-03-07-01-55-46-402`  
**Status:** Failed after ~447s (same error as previous job)

### Error

Same shape mismatch as the first job despite the fix already being committed:

```
ValueError: Trying to set a tensor of shape torch.Size([1024, 5120]) in "weight"
(which has shape torch.Size([1280, 5120])), this looks incorrect.
```

### Root Cause

SageMaker packages `source_dir` into a tarball at submission time and can reuse a cached
tarball if the contents hash to the same value. The second job was submitted immediately
after the fix was committed, and SageMaker served the cached tarball from the first
(broken) job — so the updated `train.py` was never actually used.

### Fix

Made a cosmetic comment change to `train.py` to alter the tarball checksum, forcing
SageMaker to repackage `./src` fresh on the next submission:

```python
# Before
# model_id -> (r, lora_alpha, target_modules)

# After
# model_id -> (r, lora_alpha, target_modules); Mistral-Nemo uses GQA — target all 4 proj layers
```

Then resubmitted, producing a new job:
`telco-rca-mistral-nemo-base-2407-2026-03-07-02-20-45-264`

### Prevention

Always make at least a trivial change to any file in `source_dir` before resubmitting
a SageMaker job after a code fix. Alternatively, use a unique `base_job_name` or
explicitly set `code_location` to a new S3 prefix to guarantee a fresh upload.


---

## [2026-03-07] transformers==4.36 DLC incompatible with Mistral-Nemo-Base-2407

**Job:** `telco-rca-mistral-nemo-base-2407-2026-03-07-02-20-45-264`  
**Status:** Failed after ~451s (same shape error, different root cause)

### Error

```
ValueError: Trying to set a tensor of shape torch.Size([1024, 5120]) in "weight"
(which has shape torch.Size([1280, 5120])), this looks incorrect.
```

Traceback pointed to `transformers/modeling_utils.py → _load_state_dict_into_meta_model` — the crash happened inside `AutoModelForCausalLM.from_pretrained`, before PEFT even ran.

### Root Cause

`transformers==4.36` (the originally pinned DLC version) predates Mistral-Nemo-Base-2407 (released July 2024). The older transformers loaded an incorrect architecture config for the model, causing a weight shape mismatch when loading the checkpoint into the meta model. This was misidentified initially as a PEFT/LoRA target module issue.

### Fix

Upgraded the SageMaker DLC to the next available version that supports Mistral-Nemo and has a valid image:

| Parameter | Before | After |
|---|---|---|
| `transformers_version` | `4.36` | `4.46.1` |
| `pytorch_version` | `2.1` | `2.3.0` |
| `py_version` | `py310` | `py311` |

DLC image used: `763104351884.dkr.ecr.us-east-1.amazonaws.com/huggingface-pytorch-training:2.3.0-transformers4.46.1-gpu-py311-cu121-ubuntu20.04`

> Note: The `pytorch_version="2.1"` pin in the README was intended to avoid a CUBLAS regression in `torch 2.10+cu128`. That regression does not affect `2.3.0+cu121`, so the upgrade is safe.

New job submitted: `telco-rca-mistral-nemo-base-2407-2026-03-07-02-45-53-116`


---

## [2026-03-07] Meta tensor offloading breaks PEFT on single-GPU BF16 LoRA

**Job:** `telco-rca-mistral-nemo-base-2407-2026-03-07-02-45-53-116`  
**Status:** Failed after ~577s

### Error

```
NotImplementedError: Cannot copy out of meta tensor; no data!
Please use torch.nn.Module.to_empty() instead of torch.nn.Module.to()
when moving module from meta to a different device.
```

Log also showed: `Some parameters are on the meta device because they were offloaded to the cpu.`

### Root Cause

`device_map="auto"` on a single `ml.g5.2xlarge` (24GB VRAM) caused transformers to split
Mistral-Nemo (12B BF16 ≈ 24GB) across GPU and CPU, placing some layers on the meta device.
When PEFT's `get_peft_model()` subsequently tried to wrap those layers, it hit a PyTorch
restriction that prevents copying data out of meta tensors.

### Fix

For BF16 LoRA (non-quantized), replaced `device_map="auto"` with `device_map={"": 0}` to
force all layers onto `cuda:0`. For QLoRA (4-bit), `device_map="auto"` is still required
by bitsandbytes and is safe because 4-bit weights fit comfortably in VRAM.

```python
# Before
model = AutoModelForCausalLM.from_pretrained(
    args.model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto",   # splits across CPU+GPU, breaks PEFT
    ...
)

# After
load_kwargs["device_map"] = {"": 0}   # BF16 LoRA: all layers on cuda:0
# load_kwargs["device_map"] = "auto"  # QLoRA only
```

New job submitted: `telco-rca-mistral-nemo-base-2407-2026-03-07-03-19-15-070`


---

## [2026-03-07] CUDA out of memory loading 12B BF16 model on 24GB A10G

**Job:** `telco-rca-mistral-nemo-base-2407-2026-03-07-03-19-15-070`  
**Status:** Failed after ~592s

### Error

```
torch.cuda.OutOfMemoryError: CUDA out of memory. Tried to allocate 140.00 MiB.
```

Traceback pointed to `set_module_tensor_to_device` — crash during model loading, not training.

### Root Cause

`device_map={"": 0}` (from the previous fix) forces all 12B BF16 weights (~24GB) onto
`cuda:0` in one shot. The A10G has exactly 24GB VRAM, leaving zero headroom for the CUDA
runtime, activations, or optimizer states. The 140MB allocation that tipped it over was
the final layer being moved to GPU.

### Fix

Three changes to `train.py`:

1. Load model on CPU first (`low_cpu_mem_usage=True`, no `device_map` for BF16), let PEFT
   wrap it, then move to GPU with `model.to("cuda")`. This avoids the all-at-once GPU
   allocation spike.

2. Enable gradient checkpointing (`model.gradient_checkpointing_enable()` +
   `gradient_checkpointing=True` in `TrainingArguments`). Trades recomputation for memory —
   reduces activation memory from O(layers) to O(1) during the backward pass.

3. Reduce `per_device_train_batch_size` from 4 → 1, increase
   `gradient_accumulation_steps` from 2 → 8. Effective batch size stays the same (8)
   but peak activation memory per step drops 4×.

New job submitted: `telco-rca-mistral-nemo-base-2407-2026-03-07-04-06-05-439`


---

## [2026-03-07] CUDA OOM during model loading — Mistral-Nemo BF16 doesn't fit on single A10G

**Job:** `telco-rca-mistral-nemo-base-2407-2026-03-07-03-19-15-070`  
**Status:** Failed after ~592s

### Error

```
torch.cuda.OutOfMemoryError: CUDA out of memory. Tried to allocate 140.00 MiB.
```

Crash occurred at shard 4/5 during `AutoModelForCausalLM.from_pretrained` inside
`_load_state_dict_into_meta_model → set_module_tensor_to_device`.

### Root Cause

Mistral-Nemo-Base-2407 at BF16 is 12B params × 2 bytes ≈ 24GB — exactly the A10G VRAM
limit. Loading directly to `cuda:0` leaves zero headroom for the remaining shards,
activations, or optimizer states, causing OOM mid-load.

### Fix

Changed to `device_map="cpu"` for BF16 LoRA to force the full model onto CPU RAM first.
After PEFT wrapping on CPU, the model is moved to GPU with gradient checkpointing enabled
to keep training-time VRAM usage within the 24GB limit.

```python
# Before
load_kwargs["device_map"] = {"": 0}  # OOM — 24GB model on 24GB GPU leaves no headroom

# After
load_kwargs["device_map"] = "cpu"    # load on CPU → PEFT wrap → move to GPU
# then after get_peft_model():
model = model.to("cuda")
model.gradient_checkpointing_enable()
```

New job submitted: `telco-rca-mistral-nemo-base-2407-2026-03-07-14-53-48-076`
