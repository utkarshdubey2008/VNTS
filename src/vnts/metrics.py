"""
VYOMA Research Initiative: Performance and Resource Measurement Suite
Tracks loss, accuracy/R2, peak memory (RSS MB), wall-clock time, and computational FLOPs.
"""

import time
import psutil
import os
import torch
import numpy as np

class ExperimentTracker:
    def __init__(self, name):
        self.name = name
        self.process = psutil.Process(os.getpid())
        self.start_time = None
        self.total_time = 0.0
        self.peak_rss_mb = 0.0
        self.initial_rss_mb = self.get_current_rss_mb()
        self.step_times = []
        self.train_losses = []
        self.test_losses = []
        self.test_metrics = []
        self.step_flops = []

    def get_current_rss_mb(self):
        return self.process.memory_info().rss / (1024 * 1024)

    def start(self):
        self.initial_rss_mb = self.get_current_rss_mb()
        self.peak_rss_mb = self.initial_rss_mb
        self.start_time = time.perf_counter()

    def update_memory(self):
        curr = self.get_current_rss_mb()
        if curr > self.peak_rss_mb:
            self.peak_rss_mb = curr

    def record_step(self, train_loss, flops=0):
        self.train_losses.append(train_loss)
        self.step_flops.append(flops)
        self.update_memory()

    def record_eval(self, test_loss, test_metric):
        self.test_losses.append(test_loss)
        self.test_metrics.append(test_metric)
        self.update_memory()

    def stop(self):
        self.total_time = time.perf_counter() - self.start_time
        self.update_memory()

    def summary(self):
        net_memory_mb = max(0.0, self.peak_rss_mb - self.initial_rss_mb)
        return {
            "name": self.name,
            "total_time_s": round(self.total_time, 4),
            "initial_rss_mb": round(self.initial_rss_mb, 2),
            "peak_rss_mb": round(self.peak_rss_mb, 2),
            "net_memory_mb": round(net_memory_mb, 2),
            "final_train_loss": round(self.train_losses[-1], 6) if self.train_losses else None,
            "final_test_loss": round(self.test_losses[-1], 6) if self.test_losses else None,
            "final_test_metric": round(self.test_metrics[-1], 6) if self.test_metrics else None,
            "total_steps": len(self.train_losses),
            "total_flops": sum(self.step_flops)
        }


def compute_classification_accuracy(preds, targets):
    pred_labels = preds.argmax(dim=-1)
    return (pred_labels == targets).float().mean().item()


def compute_r2_score(preds, targets):
    preds_np = preds.detach().cpu().numpy()
    targets_np = targets.detach().cpu().numpy()
    ss_res = np.sum((targets_np - preds_np)**2)
    ss_tot = np.sum((targets_np - np.mean(targets_np))**2)
    if ss_tot == 0:
        return 0.0
    return float(1.0 - (ss_res / ss_tot))
