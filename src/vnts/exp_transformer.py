"""
VNTS Experiment: Autoregressive Sequence Modeling on TinyShakespeare
Evaluates Standard Causal Transformer (Backpropagation) vs
Decoupled Forward Transformer (DFT / FRP-Transformer).
"""

import os
import json
import time
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from vnts.transformer_models import StandardTransformerLM, DecoupledForwardTransformerLM
from vnts.metrics import ExperimentTracker

def get_text_data(path="/root/vnts_clone/data/tinyshakespeare.txt"):
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()
    chars = sorted(list(set(text)))
    vocab_size = len(chars)
    char2idx = {ch: i for i, ch in enumerate(chars)}
    idx2char = {i: ch for i, ch in enumerate(chars)}
    data = torch.tensor([char2idx[ch] for ch in text], dtype=torch.long)
    
    # 90% train, 10% val
    n = int(0.9 * len(data))
    train_data = data[:n]
    val_data = data[n:]
    return train_data, val_data, vocab_size, char2idx, idx2char


def get_batch(data, batch_size=32, seq_len=64):
    ix = torch.randint(len(data) - seq_len, (batch_size,))
    x = torch.stack([data[i:i+seq_len] for i in ix])
    y = torch.stack([data[i+1:i+seq_len+1] for i in ix])
    return x, y


def train_standard_transformer(train_data, val_data, vocab_size, steps=600, batch_size=32, seq_len=64, lr=3e-3):
    print("\n---> [1/2] Training Standard Causal Transformer (Backprop + AdamW) ...", flush=True)
    model = StandardTransformerLM(vocab_size=vocab_size, d_model=64, n_heads=4, n_layers=4, max_seq_len=seq_len)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
    tracker = ExperimentTracker("Standard-Transformer-Backprop")
    tracker.start()

    criterion = nn.CrossEntropyLoss()
    
    for step in range(steps):
        x, y = get_batch(train_data, batch_size, seq_len)
        optimizer.zero_grad()
        logits = model(x) # (B, S, V)
        loss = criterion(logits.view(-1, vocab_size), y.view(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        n_params = sum(p.numel() for p in model.parameters())
        flops = 6 * n_params * batch_size * seq_len
        tracker.record_step(loss.item(), flops)

        if (step + 1) % 150 == 0 or step == steps - 1:
            print(f"     Step {step+1:3d}/{steps}: Train Loss = {loss.item():.4f}", flush=True)

    tracker.stop()

    # Evaluate validation loss and perplexity
    model.eval()
    val_loss_total = 0.0
    val_batches = 20
    with torch.no_grad():
        for _ in range(val_batches):
            vx, vy = get_batch(val_data, batch_size, seq_len)
            v_logits = model(vx)
            val_loss_total += criterion(v_logits.view(-1, vocab_size), vy.view(-1)).item()
    val_loss = val_loss_total / val_batches
    perplexity = math.exp(min(val_loss, 20.0))

    summary = tracker.summary()
    summary["val_loss"] = round(val_loss, 4)
    summary["val_perplexity"] = round(perplexity, 2)
    summary["tokens_per_sec"] = round((steps * batch_size * seq_len) / summary["total_time_s"], 1)

    print(f"  --> Standard Backprop Done: Val Loss = {val_loss:.4f} | Perplexity = {perplexity:.2f} | Speed = {summary['tokens_per_sec']} tok/s | Peak RSS = {summary['peak_rss_mb']:.1f} MB")
    return model, summary


def train_decoupled_forward_transformer(train_data, val_data, vocab_size, steps=600, batch_size=32, seq_len=64, lr=3e-3):
    print("\n---> [2/2] Training Decoupled Forward Transformer (DFT / FRP-Transformer) ...", flush=True)
    model = DecoupledForwardTransformerLM(vocab_size=vocab_size, d_model=64, n_heads=4, n_layers=4, max_seq_len=seq_len)
    
    # Layer-wise isolated optimizers
    # Block 0 optimizes embedding + block 0 + local head 0
    emb_opt = torch.optim.AdamW(list(model.token_emb.parameters()) + list(model.pos_emb.parameters()), lr=lr)
    block_opts = [
        torch.optim.AdamW(list(block.parameters()) + list(ln.parameters()) + list(head.parameters()), lr=lr, weight_decay=1e-2)
        for block, ln, head in zip(model.blocks, model.local_lns, model.local_heads)
    ]
    final_opt = torch.optim.AdamW(list(model.ln_f.parameters()) + list(model.final_head.parameters()), lr=lr, weight_decay=1e-2)

    tracker = ExperimentTracker("Decoupled-Forward-Transformer-FRP")
    tracker.start()
    criterion = nn.CrossEntropyLoss()

    for step in range(steps):
        x, y = get_batch(train_data, batch_size, seq_len)
        B, S = x.shape
        pos = torch.arange(0, S, device=x.device).unsqueeze(0)
        
        # 1. Forward embed
        emb_opt.zero_grad()
        h0 = model.token_emb(x) + model.pos_emb(pos)
        
        # 2. Block-by-block decoupled update
        h_curr = h0
        n_params = sum(p.numel() for p in model.parameters())
        step_flops = 0

        for block_idx, (block, ln, head, opt) in enumerate(zip(model.blocks, model.local_lns, model.local_heads, block_opts)):
            opt.zero_grad()
            # Input is detached from preceding block
            h_in = h_curr.detach()
            h_out = block(h_in)
            
            # Local token logits and loss
            logits_local = head(ln(h_out))
            loss_local = criterion(logits_local.view(-1, vocab_size), y.view(-1))
            
            # Update current block immediately
            loss_local.backward()
            opt.step()
            
            # If first block, also update embeddings using first block error
            if block_idx == 0:
                h0_loss = criterion(head(ln(block(h0))).view(-1, vocab_size), y.view(-1))
                h0_loss.backward()
                emb_opt.step()

            # Detach activations immediately to free backward graph and attention maps
            h_curr = h_out.detach()
            step_flops += 4 * (sum(p.numel() for p in block.parameters()) + head.weight.numel()) * B * S

        # Final readout update
        final_opt.zero_grad()
        final_logits = model.final_head(model.ln_f(h_curr))
        final_loss = criterion(final_logits.view(-1, vocab_size), y.view(-1))
        final_loss.backward()
        final_opt.step()

        tracker.record_step(final_loss.item(), step_flops)

        if (step + 1) % 150 == 0 or step == steps - 1:
            print(f"     Step {step+1:3d}/{steps}: Train Loss = {final_loss.item():.4f}", flush=True)

    tracker.stop()

    # Evaluate validation loss and perplexity
    model.eval()
    val_loss_total = 0.0
    val_batches = 20
    with torch.no_grad():
        for _ in range(val_batches):
            vx, vy = get_batch(val_data, batch_size, seq_len)
            v_logits = model(vx)
            val_loss_total += criterion(v_logits.view(-1, vocab_size), vy.view(-1)).item()
    val_loss = val_loss_total / val_batches
    perplexity = math.exp(min(val_loss, 20.0))

    summary = tracker.summary()
    summary["val_loss"] = round(val_loss, 4)
    summary["val_perplexity"] = round(perplexity, 2)
    summary["tokens_per_sec"] = round((steps * batch_size * seq_len) / summary["total_time_s"], 1)

    print(f"  --> Decoupled DFT Done: Val Loss = {val_loss:.4f} | Perplexity = {perplexity:.2f} | Speed = {summary['tokens_per_sec']} tok/s | Peak RSS = {summary['peak_rss_mb']:.1f} MB")
    return model, summary


def generate_sample(model, prompt="ROMEO:", max_new_tokens=100, char2idx=None, idx2char=None):
    model.eval()
    idx = torch.tensor([char2idx[ch] for ch in prompt], dtype=torch.long).unsqueeze(0)
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -64:]
        with torch.no_grad():
            logits = model(idx_cond)
            logits = logits[:, -1, :] / 0.8
            probs = F.softmax(logits, dim=-1)
            next_idx = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, next_idx), dim=1)
    out_text = "".join([idx2char[i.item()] for i in idx[0]])
    return out_text


def run_transformer_experiment(steps=600):
    print("=" * 65)
    print("  EXPERIMENT 3: AUTOREGRESSIVE SEQUENCE MODELING (TRANSFORMERS)")
    print("=" * 65)

    train_data, val_data, vocab_size, char2idx, idx2char = get_text_data()
    print(f"Dataset loaded: {len(train_data)+len(val_data):,} characters | Vocab size: {vocab_size}")

    bp_model, bp_summary = train_standard_transformer(train_data, val_data, vocab_size, steps=steps)
    dft_model, dft_summary = train_decoupled_forward_transformer(train_data, val_data, vocab_size, steps=steps)

    print("\n--- SAMPLE GENERATION TEST ---")
    bp_sample = generate_sample(bp_model, "ROMEO: ", 80, char2idx, idx2char)
    dft_sample = generate_sample(dft_model, "ROMEO: ", 80, char2idx, idx2char)
    print(f"\n[Standard Backprop Sample]:\n{bp_sample}")
    print(f"\n[Decoupled Forward Transformer (DFT) Sample]:\n{dft_sample}")

    results = {
        "standard_backprop": bp_summary,
        "decoupled_forward_transformer": dft_summary,
        "samples": {
            "backprop": bp_sample,
            "dft": dft_sample
        }
    }

    out_file = "/root/vnts_clone/results/exp_transformer.json"
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nTransformer experiment results saved to {out_file}")
    return results

if __name__ == "__main__":
    run_transformer_experiment()
