"""
Phase 6 Benchmark: Edge Deployment & Micro-LLM Language Modeling
Compares Micro-LLM (Multi-Head COMT) vs PyTorch RNN/LSTM Baseline on commodity CPU.
"""

import os
import time
import json
import tracemalloc
import torch
import torch.nn as nn
import torch.optim as optim

from vnts import MicroLLM


def run_edge_llm_benchmark():
    print("=" * 70)
    print("Phase 6: Micro-LLM Language Modeling on Commodity Edge Hardware")
    print("=" * 70)

    # Realistic literary sample corpus
    corpus = """
    To be, or not to be, that is the question:
    Whether 'tis nobler in the mind to suffer
    The slings and arrows of outrageous fortune,
    Or to take arms against a sea of troubles
    And by opposing end them. To die—to sleep,
    No more; and by a sleep to say we end
    The heart-ache and the thousand natural shocks
    That flesh is heir to: 'tis a consummation
    Devoutly to be wish'd. To die, to sleep;
    To sleep, perchance to dream—ay, there's the rub:
    For in that sleep of death what dreams may come,
    When we have shuffled off this mortal coil,
    Must give us pause. There's the respect
    That makes calamity of so long life.
    """ * 10  # 10x repetitions for training

    print(f"Corpus size: {len(corpus)} characters")

    # 1. Micro-LLM (Multi-Head COMT)
    tracemalloc.start()
    t0 = time.perf_counter()

    model = MicroLLM.from_text(corpus, embed_dim=128, num_heads=4)
    vnts_stats = model.train_corpus(corpus, seq_len=32)

    vnts_time = time.perf_counter() - t0
    _, peak_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    vnts_peak_mb = peak_bytes / (1024 * 1024)

    prompt = "To be, or not"
    generated_text = model.generate(prompt=prompt, max_new_tokens=60, temperature=0.7)

    print(f"\n[Micro-LLM]")
    print(f"Training Time: {vnts_time:.4f}s")
    print(f"Next-Token Accuracy: {vnts_stats['next_token_accuracy'] * 100:.2f}%")
    print(f"Throughput: {vnts_stats['throughput_tokens_per_s']:.1f} tokens/s")
    print(f"Peak Allocated RAM: {vnts_peak_mb:.3f} MB")
    print(f"Generated Sample: {repr(generated_text[:80])}...")

    # 2. PyTorch Baseline (LSTM with BPTT & AdamW)
    token_ids = model.encode_text(corpus)
    vocab_size = model.vocab_size

    class BaselineLSTM(nn.Module):
        def __init__(self, vocab_size, embed_dim=128):
            super().__init__()
            self.emb = nn.Embedding(vocab_size, embed_dim)
            self.lstm = nn.LSTM(embed_dim, embed_dim, batch_first=True)
            self.fc = nn.Linear(embed_dim, vocab_size)

        def forward(self, x):
            e = self.emb(x)
            out, _ = self.lstm(e)
            return self.fc(out)

    torch.manual_seed(42)
    py_model = BaselineLSTM(vocab_size, embed_dim=128)
    optimizer = optim.AdamW(py_model.parameters(), lr=2e-3)
    criterion = nn.CrossEntropyLoss()

    seq_len = 32
    num_samples = (len(token_ids) - 1) // seq_len
    chunks = [token_ids[i * seq_len : i * seq_len + seq_len + 1] for i in range(num_samples)]
    batch = torch.stack(chunks)
    X_pt = batch[:, :-1]
    Y_pt = batch[:, 1:]

    tracemalloc.start()
    t0_pt = time.perf_counter()

    py_model.train()
    # Train for 20 epochs
    for ep in range(20):
        optimizer.zero_grad()
        logits = py_model(X_pt)
        loss = criterion(logits.reshape(-1, vocab_size), Y_pt.reshape(-1))
        loss.backward()
        optimizer.step()

    pt_time = time.perf_counter() - t0_pt
    _, peak_bytes_pt = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    pt_peak_mb = peak_bytes_pt / (1024 * 1024)

    py_model.eval()
    with torch.no_grad():
        pt_preds = torch.argmax(py_model(X_pt), dim=-1)
        pt_acc = (pt_preds == Y_pt).float().mean().item()

    print(f"\n[PyTorch LSTM (BPTT + AdamW)]")
    print(f"Training Time: {pt_time:.4f}s")
    print(f"Next-Token Accuracy: {pt_acc * 100:.2f}%")
    print(f"Peak Allocated RAM: {pt_peak_mb:.3f} MB")
    print(f"Speedup: {pt_time / max(1e-6, vnts_time):.2f}x faster with Micro-LLM")

    results = {
        "vnts_micro_llm": {
            "training_time_s": vnts_time,
            "next_token_accuracy": vnts_stats["next_token_accuracy"],
            "throughput_tokens_per_s": vnts_stats["throughput_tokens_per_s"],
            "peak_allocated_mb": vnts_peak_mb,
            "generated_sample": generated_text
        },
        "pytorch_lstm": {
            "training_time_s": pt_time,
            "next_token_accuracy": pt_acc,
            "peak_allocated_mb": pt_peak_mb
        },
        "speedup_factor": pt_time / max(1e-6, vnts_time)
    }

    os.makedirs("results", exist_ok=True)
    with open("results/phase6_edge_llm_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print("\n" + "=" * 70)
    print("Benchmark complete. Results saved -> results/phase6_edge_llm_results.json")
    print("=" * 70)


if __name__ == "__main__":
    run_edge_llm_benchmark()
