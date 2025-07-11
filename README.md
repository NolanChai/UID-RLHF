# UID RLHF
This is a draft environment for the full UID package, which is a multi-purpose package for our paper, offering features including:
- Calculating the UID of a given text using an LLM tokenizer
- Integration with Huggingface
- Speedups for inference

# Getting Started
To get started, use uv to install and sync requirements. You can do this by running:
```
pip install uv
```
and run `uv sync` after changing directory to this folder. 

You can either run this as a cli through `main.py`, or run as a package. There are examples in `sandbox.ipynb`.

To launch a jupyterlab server with our virtual environment, run
```
uv run --with jupyter jupyter lab
```