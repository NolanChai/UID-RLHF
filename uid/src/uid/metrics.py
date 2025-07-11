from .spinner import *
import torch
import numpy as np
import torch.nn.functional as F
from transformers import AutoTokenizer as HFTokenizer, AutoModelForCausalLM
from transformers.utils.logging import disable_progress_bar
from huggingface_hub import snapshot_download
from tqdm.rich import tqdm as _tqdm_rich

from rich.progress import (
    SpinnerColumn,
    TextColumn,
    BarColumn,
    DownloadColumn,
    TransferSpeedColumn,
    TimeRemainingColumn,
)

import warnings
from tqdm import TqdmExperimentalWarning
warnings.filterwarnings("ignore", category=TqdmExperimentalWarning)

# Disable symlinks warning
import os
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

class TqdmRichBar(_tqdm_rich):
    def __init__(self, *args, **kwargs):
        # If someone passed a custom progress spec, respect it; otherwise use ours:
        if "progress" not in kwargs:
            kwargs["progress"] = (
                SpinnerColumn(), 
                TextColumn("[progress.description]{task.description}"), 
                BarColumn(), 
                TextColumn("{task.percentage:>3.0f}%"),
                # DownloadColumn(), 
                # TransferSpeedColumn(),
                TimeRemainingColumn()
            )
        super().__init__(*args, **kwargs)

class Tokenizer:
    def __init__(self, model_name: str = "gpt2", device: str = None):
        """
            Arguments:
            -   model_name = Huggingface model identifier for a causal LM
            -   device = GPU, only supports cuda or cpu atm
        """
        # Disable huggingface progress bars
        disable_progress_bar()

        repo_dir = snapshot_download(
            repo_id=model_name,
            local_files_only=False,
            tqdm_class=TqdmRichBar,
        )

        # Fetch tokenizer from Huggingface
        self.tokenizer = HFTokenizer.from_pretrained(model_name, local_files_only = True)

        # Sets padding to EOS by default if missing a specific one
        if self.tokenizer.pad_token == None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(model_name)

        # Load from a local snapshot (cached download)
        self.tokenizer = HFTokenizer.from_pretrained(repo_dir, local_files_only=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(repo_dir, local_files_only=True)

        # Place on device
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()

    def surprisal(self, text: str) -> list[float]:
        """
        Compute the surprisal of each token in the text using the tokenizer.

        Returns a list of surprisal values, indexed by token. 
        """

        # Tokenize
        enc = self.tokenizer(text, return_tensors="pt")
        input_ids = enc["input_ids"].to(self.device)

        # Forward pass with labels to get loss per token
        with torch.no_grad():
            outputs = self.model(input_ids, labels=input_ids)
            # Logits are in the format: [batch, seq_len, vocab_size]
            logits = outputs.logits

        # Shift logits and labels for next token prediction
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = input_ids[:, 1:].contiguous()

        # Compute log probabilities
        log_probs = F.log_softmax(shift_logits, dim=-1) # Natural log

        # Log probabilities of each true next token
        # This is in the format of [batch, seq_len-1]
        token_log_probs = log_probs.gather(-1, shift_labels.unsqueeze(-1)).squeeze(-1)

        # Convert to surprisal in bits using: -log₂(p) = -ln(p) / ln(2)
        surprisal_bits = (-token_log_probs / torch.log(torch.tensor(2.0))).squeeze(0)

        # Pad the first token with None or 0.0 (no context)
        return [None] + surprisal_bits.cpu().tolist()
