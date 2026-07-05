# Local LoRAs

Place local LoRA `.safetensors` files in this folder. The GUI scans this folder automatically and shows discovered adapters by filename stem.

The CLI accepts either a discovered catalog id or a direct path:

```bash
.venv/bin/krea-2-turbo-mlx generate \
  --model artifacts/krea-2-turbo-mlx \
  --prompt "portrait lighting study" \
  --lora my-style:0.8
```

Use `--lora-dir PATH` to scan a different folder. Use `--lora ID_OR_PATH[:SCALE]` to apply one or more adapters. Scale must be from `0.0` to `4.0`; the default is `1.0`.

Supported adapter families are standard LoRA, weight-diff LoRA, and LoKr. Unsupported tensor families are skipped with warnings when possible.

LoRA weight files are ignored by git so they are not accidentally shared from this repository.
