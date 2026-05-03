import argparse
import sys
import os

# Ensure repo root is on sys.path whether this file is imported or run directly
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

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

    def __init__(self, neurolm: NeuroLM, pool_layer: int = -1,
                 num_chans: int = 19, pool_time: bool = True,
                 pool_chans: bool = True):
        """
        Args:
            pool_time:  Pool over the time dimension.
            pool_chans: Pool over the channel dimension.
            num_chans:  Channels per time step (default 19).

        Output shape by combination:
            pool_time=True,  pool_chans=True  → (B, D)
            pool_time=False, pool_chans=True  → (B, num_time, D)
            pool_time=True,  pool_chans=False → (B, num_chans, D)
            pool_time=False, pool_chans=False → (B, num_time, num_chans, D)

        Invalid positions (eeg_mask=False) are zeroed out in all cases.
        """
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
        self.pool_time = pool_time
        self.pool_chans = pool_chans
        self.num_chans = num_chans

    def forward(
        self,
        x_eeg,        # (B, N_eeg, PATCH_SIZE)  float
        input_chans,  # (B, N_eeg)               int   — standard_1020 indices
        input_time,   # (B, N_eeg)               int   — time-step indices
        input_mask,   # (B, N_eeg)               bool  — valid patch mask for tokenizer
        eeg_mask,     # (B, N_eeg)               bool  — True = valid token
    ):
        gpt = self.neurolm.GPT2

        # EEG tokenization: raw patches → VQ indices → projected embeddings
        tok_mask = input_mask.unsqueeze(1).repeat(1, x_eeg.size(1), 1).unsqueeze(1)
        x = self.neurolm.tokenizer(
            x_eeg, input_chans, input_time, tok_mask, return_all_tokens=True
        )
        x = self.neurolm.encode_transform_layer(x)
        x = x + self.neurolm.pos_embed(input_chans)
        x = x + gpt.transformer.wpe(input_time)

        attn_mask = eeg_mask[:, None, None, :]       # (B, 1, 1, N_eeg) key mask
        x = gpt.transformer.drop(x)
        for block in gpt.transformer.h[:self.pool_layer + 1]:
            x = block(x, attn_mask)

        if self.apply_ln_f:
            x = gpt.transformer.ln_f(x)

        # Reshape flat sequence → (B, num_time, num_chans, D)
        # Tokens are time-major: [t0_c0..t0_c18, t1_c0..t1_c18, ...]
        B, N, D = x.shape
        num_time = N // self.num_chans
        x = x.view(B, num_time, self.num_chans, D)
        w = eeg_mask.float().view(B, num_time, self.num_chans, 1)

        # Zero out invalid positions before any reduction
        x = x * w

        if self.pool_time and self.pool_chans:
            # (B, D)
            return x.sum(dim=(1, 2)) / w.sum(dim=(1, 2)).clamp(min=1.0)
        elif not self.pool_time and self.pool_chans:
            # (B, num_time, D)
            return x.sum(dim=2) / w.sum(dim=2).clamp(min=1.0)
        elif self.pool_time and not self.pool_chans:
            # (B, num_chans, D)
            return x.sum(dim=1) / w.sum(dim=1).clamp(min=1.0)
        else:
            # (B, num_time, num_chans, D) — already zeroed at invalid positions
            return x


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', default='/home/alimirz/2026/neurolm/NeuroLM-B.pt', help='Path to NeuroLM-B.pt checkpoint')
    parser.add_argument('--pool_layer', type=int, default=-1)
    parser.add_argument('--batch_size', type=int, default=2)
    args = parser.parse_args()

    from model.model import GPTConfig
    from dataset import standard_1020

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")

    # Load checkpoint
    print(f"Loading {args.ckpt} ...")
    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
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

    cases = [
        dict(pool_time=True,  pool_chans=True),   # (B, D)
        dict(pool_time=False, pool_chans=True),   # (B, T, D)
        dict(pool_time=True,  pool_chans=False),  # (B, C, D)
        dict(pool_time=False, pool_chans=False),  # (B, T, C, D)
    ]
    for kw in cases:
        enc = NeuroLMEncoder(neurolm, pool_layer=args.pool_layer,
                             num_chans=NUM_CHANS, **kw).to(device)
        with torch.no_grad():
            out = enc(x_eeg, input_chans, input_time, input_mask, eeg_mask)
        print(f"pool_time={str(kw['pool_time']):5s}  pool_chans={str(kw['pool_chans']):5s}  → {tuple(out.shape)}")
    print("OK")
