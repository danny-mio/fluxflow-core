"""Benchmark script comparing BezierActivation/TrainableBezier against
PadeActivation/TrainablePade -- forward+backward speed and parameter count."""

import time

import torch

from fluxflow.models.activations import BezierActivation, TrainableBezier
from fluxflow.models.pade_activation import PadeActivation, TrainablePade


def benchmark_activation(activation, x, name, warmup=10, iterations=100):
    for _ in range(warmup):
        _ = activation(x)
    if x.is_cuda:
        torch.cuda.synchronize()

    start = time.time()
    output = None
    for _ in range(iterations):
        output = activation(x)
    if x.is_cuda:
        torch.cuda.synchronize()

    elapsed = time.time() - start
    avg_time = (elapsed / iterations) * 1000

    assert output is not None
    print(f"{name:30s} | {avg_time:8.4f} ms/iter | {output.shape}")
    return avg_time


def benchmark_forward_backward(activation, x, name):
    x = x.clone().requires_grad_(True)
    start = time.time()
    output = activation(x)
    loss = output.sum()
    loss.backward()
    if x.is_cuda:
        torch.cuda.synchronize()
    elapsed = (time.time() - start) * 1000
    print(f"{name:30s} | {elapsed:8.4f} ms (fwd+bwd)")


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Running Bezier vs Padé benchmarks on: {device}")
    print("=" * 70)

    bezier_fixed = BezierActivation().to(device)
    pade_fixed = PadeActivation().to(device)

    test_cases = [
        ("2D Small", (4, 50)),
        ("2D Medium", (32, 250)),
        ("2D Large", (64, 1000)),
        ("3D Small", (4, 64, 125)),
        ("3D Medium", (8, 128, 250)),
        ("4D Small", (4, 15, 8, 8)),
        ("4D Large", (16, 60, 32, 32)),
    ]

    print("\n--- Fixed / data-driven mode (BezierActivation vs PadeActivation) ---")
    print(f"\n{'Test Case':<30} | {'Time (ms)':<12} | Output Shape")
    print("-" * 70)
    for name, shape in test_cases:
        x = torch.randn(*shape, device=device)
        benchmark_activation(bezier_fixed, x, f"Bezier {name}")
        benchmark_activation(pade_fixed, x, f"Pade {name}")

    print("\n--- Trainable mode (TrainableBezier vs TrainablePade), shape (128,) ---")
    bezier_trainable = TrainableBezier((128,)).to(device)
    pade_trainable = TrainablePade((128,)).to(device)
    x = torch.randn(32, 128, device=device)
    benchmark_activation(bezier_trainable, x, "TrainableBezier")
    benchmark_activation(pade_trainable, x, "TrainablePade")

    print("\n--- Forward + backward pass ---")
    x = torch.randn(8, 125, device=device)
    benchmark_forward_backward(bezier_fixed, x, "Bezier fwd+bwd")
    benchmark_forward_backward(pade_fixed, x, "Pade fwd+bwd")

    print("\n--- Parameter counts ---")
    bezier_params = sum(p.numel() for p in TrainableBezier((128,)).parameters())
    pade_params = sum(p.numel() for p in TrainablePade((128,)).parameters())
    print(f"TrainableBezier params (D=128): {bezier_params}")
    print(f"TrainablePade params (D=128):   {pade_params}")

    print("\n✓ All benchmarks completed successfully!")


if __name__ == "__main__":
    main()
