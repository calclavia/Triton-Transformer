# Triton Transformer
[WIP]

Implementations of some expensive Transformer components in [OpenAI Triton](https://github.com/openai/triton) JIT for Pytorch.

## Running
You must have an NVIDIA GPU in order to run this.

Running in Docker:
```
docker run --gpus=all -v $(pwd):/workspace --rm -it pytorch/pytorch:1.10.0-cuda11.3-cudnn8-devel /bin/bash

pip install -r requirements.txt
```

Run tests:
```
python -m unittest
```

Run benchmark:
```
python -m ttx.attention.benchmark
```

To run test against Fast Transformers, install the following dependency:
```
pip install pytorch-fast-transformers
```