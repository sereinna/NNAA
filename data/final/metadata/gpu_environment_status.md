# GPU Environment Status

Date: 2026-06-29

Conda environment:

```text
nnaa
```

PyTorch GPU installation:

```text
torch 2.11.0+cu128
torchvision 0.26.0+cu128
torchaudio 2.11.0+cu128
```

CUDA status:

```text
torch.cuda.is_available(): True
torch.version.cuda: 12.8
device_count: 2
device0: NVIDIA GeForce RTX 5090
```

Smoke test:

```text
matrix multiply + backward on cuda:0 passed
```

Install command used:

```bash
conda run -n nnaa python -m pip install \
  --index-url https://download.pytorch.org/whl/cu128 \
  torch torchvision torchaudio
```

