import argparse
import sys
import os

import torch
import torch.nn as nn

from model.model_neurolm import NeuroLM


class NeuroLMEncoder(nn.Module):
    """
    Wraps NeuroLM as a pooled EEG encoder for downstream probing.

    Runs EEG tokenization and the first `pool_layer+1` transformer blocks,
    then mean-pools hidden states over valid EEG token positions.

    Args:
        neurolm:    A loaded NeuroLM instance (weights already set).
        pool_layer: Transformer layer whose output to pool. 0-indexed;
                    -1 (default) means the final layer with ln_f applied.

    The NeuroLM weights are NOT frozen by this wrapper. For probing, freeze
    them before use:
        for p in encoder.neurolm.parameters():
            p.requires_grad_(False)
    """

    def __init__(self, neurolm: NeuroLM, pool_layer: int = -1):
        super().__init__()
        n_layers = neurolm.GPT2.config.n_layer
        if pool_layer < 0:
            pool_layer = n_layers + pool_layer
        assert 0 <= pool_layer < n_layers, \
            f"pool_layer {pool_layer} out of range for {n_layers}-layer model"

        self.neurolm = neurolm
        self.pool_layer = pool_layer
        self.apply_ln_f = (pool_layer == n_layers - 1)
        self.hidden_size = neurolm.GPT2.config.n_embd

    def forward(
        self,
        x_eeg,        # (B, N_eeg, PATCH_SIZE)  float
        input_chans,  # (B, N_eeg)               int   — standard_1020 indices
        input_time,   # (B, N_eeg)               int   — time-step indices
        input_mask,   # (B, N_eeg)               bool  — valid patch mask for tokenizer
        eeg_mask,     # (B, N_eeg)               bool  — True = valid token; used for attention and pooling
    ):
        """
        Returns:
            pooled: (B, hidden_size) mean-pooled EEG representation.
        """
        gpt = self.neurolm.GPT2

        # EEG tokenization: raw patches → VQ indices → projected embeddings
        tok_mask = input_mask.unsqueeze(1).repeat(1, x_eeg.size(1), 1).unsqueeze(1)
        x = self.neurolm.tokenizer(
            x_eeg, input_chans, input_time, tok_mask, return_all_tokens=True
        )
        x = self.neurolm.encode_transform_layer(x)
        x = x + self.neurolm.pos_embed(input_chans)

        # Add GPT2 positional embeddings (time-step positions)
        x = x + gpt.transformer.wpe(input_time)

        # Run transformer blocks up to pool_layer.
        # eeg_mask (B, N_eeg) bool acts as a key mask: invalid positions are
        # excluded from attention across all queries.
        x = gpt.transformer.drop(x)
        for block in gpt.transformer.h[:self.pool_layer + 1]:
            x = block(x, eeg_mask)

        if self.apply_ln_f:
            x = gpt.transformer.ln_f(x)

        # Masked mean pool over valid positions only
        w = eeg_mask.float().unsqueeze(-1)          # (B, N_eeg, 1)
        pooled = (x * w).sum(dim=1) / w.sum(dim=1).clamp(min=1.0)
        return pooled                               # (B, hidden_size)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', default='/home/alimirz/2026/neurolm/NeuroLM-B.pt', help='Path to NeuroLM-B.pt checkpoint')
    parser.add_argument('--pool_layer', type=int, default=-1)
    parser.add_argument('--batch_size', type=int, default=2)
    args = parser.parse_args()

    # Must run from repo root so model.* imports resolve
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from model.model import GPTConfig
    from dataset import standard_1020

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")

    # Load checkpoint
    print(f"Loading {args.ckpt} ...")
    ckpt = torch.load(args.ckpt, map_location=device)
    model_args = ckpt['model_args']
    gptconf = GPTConfig(**model_args)
    neurolm = NeuroLM(gptconf, init_from='scratch')
    state = ckpt['model']
    state = {(k[len('_orig_mod.'):] if k.startswith('_orig_mod.') else k): v
             for k, v in state.items()}
    neurolm.load_state_dict(state)
    neurolm.to(device).eval()
    for p in neurolm.parameters():
        p.requires_grad_(False)
    print("Checkpoint loaded.")

    # Build encoder
    encoder = NeuroLMEncoder(neurolm, pool_layer=args.pool_layer).to(device)
    print(f"Encoder: pool_layer={encoder.pool_layer}, "
          f"apply_ln_f={encoder.apply_ln_f}, hidden_size={encoder.hidden_size}")

    # Synthetic 30s x 19-channel EEG batch
    CHANNEL_ORDER = ['O1','O2','T6','P4','PZ','P3','T5','T3',
                     'C3','CZ','C4','T4','F8','F4','FZ','F3','F7','FP1','FP2']
    NUM_CHANS, NUM_TIME, PATCH_SIZE = 19, 30, 200
    N_EEG = NUM_TIME * NUM_CHANS  # 570

    chan_indices = [standard_1020.index(ch) for ch in CHANNEL_ORDER] * NUM_TIME
    time_indices = [t for t in range(NUM_TIME) for _ in range(NUM_CHANS)]

    B = args.batch_size
    x_eeg      = torch.randn(B, N_EEG, PATCH_SIZE, device=device)
    input_chans = torch.tensor(chan_indices, dtype=torch.int).unsqueeze(0).expand(B, -1).to(device)
    input_time  = torch.tensor(time_indices, dtype=torch.int).unsqueeze(0).expand(B, -1).to(device)
    input_mask  = torch.ones(B, N_EEG, dtype=torch.bool, device=device)
    eeg_mask    = torch.ones(B, N_EEG, dtype=torch.bool, device=device)

    # Run encoder
    with torch.no_grad():
        pooled = encoder(x_eeg, input_chans, input_time, input_mask, eeg_mask)

    print(f"Output shape: {pooled.shape}")          # expect (B, 768)
    print(f"Output mean:  {pooled.mean().item():.4f}")
    print(f"Output std:   {pooled.std().item():.4f}")
    print("OK")
