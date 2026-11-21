"""
Enterprise VNTS Benchmark Suite:
1. MORA Attention vs Transformer Multi-Head Self-Attention (Memory & Length Scaling)
2. Continuous Time-Series Forecasting: VNTS-TS vs PyTorch LSTM
3. Deep Residual Vision (Res-VNTS) Depth Scaling
"""

import os
import time
import json
import tracemalloc
import torch
import torch.nn as nn
import torch.optim as optim

from vnts.nn import MultiScaleOrthogonalAttention, ContinuousStateSpaceTracker, ResVNTSVision


def benchmark_mora_vs_transformer_attention():
    print("--- Benchmark 1: MORA vs Transformer Self-Attention (Length Scaling) ---")
    embed_dim = 128
    num_heads = 4
    batch_size = 4
    seq_lengths = [64, 128, 256, 512, 1024]

    mora_times = []
    transformer_times = []
    mora_memories = []
    transformer_memories = []

    mora = MultiScaleOrthogonalAttention(embed_dim=embed_dim, num_heads=num_heads)
    transformer_attn = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, batch_first=True)

    for T in seq_lengths:
        x = torch.randn(batch_size, T, embed_dim)

        # 1. MORA Attention
        tracemalloc.start()
        t0 = time.perf_counter()
        out_mora, _ = mora(x)
        mora_t = time.perf_counter() - t0
        _, peak_mora = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        # 2. Transformer Attention
        tracemalloc.start()
        t0_trans = time.perf_counter()
        out_trans, _ = transformer_attn(x, x, x)
        trans_t = time.perf_counter() - t0_trans
        _, peak_trans = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        mora_times.append(mora_t)
        transformer_times.append(trans_t)
        mora_memories.append(peak_mora / 1024.0)
        transformer_memories.append(peak_trans / 1024.0)

        print(f"SeqLen = {T:4d} | MORA: {mora_t*1000:6.2f} ms ({peak_mora/1024:6.1f} KB) | Transformer: {trans_t*1000:6.2f} ms ({peak_trans/1024:6.1f} KB)")

    return {
        "seq_lengths": seq_lengths,
        "mora_times_s": mora_times,
        "transformer_times_s": transformer_times,
        "mora_peak_kb": mora_memories,
        "transformer_peak_kb": transformer_memories
    }


def benchmark_time_series_forecasting():
    print("\n--- Benchmark 2: Continuous Time-Series Forecasting (VNTS-TS vs LSTM) ---")
    torch.manual_seed(42)
    B, T, D = 16, 100, 8
    # Complex non-linear multi-harmonic signal
    t = torch.linspace(0, 20, T).unsqueeze(0).repeat(B, 1)
    signals = []
    for d in range(D):
        freq = (d + 1) * 0.5
        phase = d * 0.3
        sig = torch.sin(freq * t + phase) + 0.3 * torch.cos(2 * freq * t)
        signals.append(sig)
    X_seq = torch.stack(signals, dim=-1)
    Y_target = torch.roll(X_seq, shifts=-1, dims=1)

    # 1. VNTS-TS Continuous Tracker
    t0 = time.perf_counter()
    tracker = ContinuousStateSpaceTracker(in_features=D, state_dim=64, forecast_horizon=1)
    vnts_stats = tracker.fit_sequence(X_seq, Y_target, streaming=False)
    vnts_time = time.perf_counter() - t0

    # 2. PyTorch LSTM Baseline
    class LSTMForecaster(nn.Module):
        def __init__(self, in_dim, hidden_dim):
            super().__init__()
            self.lstm = nn.LSTM(in_dim, hidden_dim, batch_first=True)
            self.fc = nn.Linear(hidden_dim, in_dim)
        def forward(self, x):
            out, _ = self.lstm(x)
            return self.fc(out)

    lstm = LSTMForecaster(in_dim=D, hidden_dim=64)
    optimizer = optim.AdamW(lstm.parameters(), lr=1e-2)
    criterion = nn.MSELoss()

    t0_lstm = time.perf_counter()
    lstm.train()
    # 25 epochs
    for ep in range(25):
        optimizer.zero_grad()
        preds = lstm(X_seq)
        loss = criterion(preds, Y_target)
        loss.backward()
        optimizer.step()
    lstm_time = time.perf_counter() - t0_lstm

    lstm.eval()
    with torch.no_grad():
        lstm_mse = criterion(lstm(X_seq), Y_target).item()

    print(f"VNTS-TS:  Time = {vnts_time:.4f}s, MSE = {vnts_stats['mse_loss']:.5f}")
    print(f"LSTM BPTT: Time = {lstm_time:.4f}s, MSE = {lstm_mse:.5f}")
    print(f"Speedup: {lstm_time / max(1e-6, vnts_time):.2f}x faster with VNTS-TS")

    return {
        "vnts_ts": {
            "time_s": vnts_time,
            "mse_loss": vnts_stats["mse_loss"]
        },
        "lstm": {
            "time_s": lstm_time,
            "mse_loss": lstm_mse
        },
        "speedup_factor": lstm_time / max(1e-6, vnts_time)
    }


def benchmark_res_vnts_depth():
    print("\n--- Benchmark 3: Deep Residual Vision (Res-VNTS) Depth Scaling ---")
    depths = [2, 4, 8, 16]
    B = 64
    images = torch.randn(B, 3, 32, 32)
    labels = torch.randint(0, 10, (B,))

    times = []
    for d in depths:
        net = ResVNTSVision(in_channels=3, num_classes=10, depth=d, features_dim=64)
        t_fit = net.assimilate(images, labels)
        times.append(t_fit)
        print(f"Depth = {d:2d} Layers | Closed-form assimilation time: {t_fit*1000:6.2f} ms")

    return {
        "depths": depths,
        "assimilation_times_s": times
    }


def main():
    print("=" * 70)
    print("Enterprise VNTS Benchmark Suite: Production-Scale Frontiers")
    print("=" * 70)

    results = {}
    results["mora_vs_transformer"] = benchmark_mora_vs_transformer_attention()
    results["time_series_forecasting"] = benchmark_time_series_forecasting()
    results["res_vnts_depth_scaling"] = benchmark_res_vnts_depth()

    os.makedirs("results", exist_ok=True)
    with open("results/vnts_enterprise_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print("\n" + "=" * 70)
    print("All enterprise benchmarks complete -> results/vnts_enterprise_results.json")
    print("=" * 70)


if __name__ == "__main__":
    main()
