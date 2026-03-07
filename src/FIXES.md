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
