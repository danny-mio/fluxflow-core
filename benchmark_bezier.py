"""Benchmark script for BezierActivation optimization."""
import torch
import time
from fluxflow.models.activations import BezierActivation


def benchmark_activation(activation, x, name, warmup=10, iterations=100):
    """Benchmark an activation function."""
    # Warmup
    for _ in range(warmup):
        _ = activation(x)
    
    # Synchronize if CUDA
    if x.is_cuda:
        torch.cuda.synchronize()
    
    # Benchmark
    start = time.time()
    output = None
    for _ in range(iterations):
        output = activation(x)
    
    if x.is_cuda:
        torch.cuda.synchronize()
    
    elapsed = time.time() - start
    avg_time = (elapsed / iterations) * 1000  # Convert to ms
    
    assert output is not None
    print(f"{name:30s} | {avg_time:8.4f} ms/iter | {output.shape}")
    return avg_time, output


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Running BezierActivation benchmarks on: {device}")
    print("=" * 70)
    
    activation = BezierActivation()
    if device == "cuda":
        activation = activation.cuda()
    
    test_cases = [
        ("2D Small", (4, 50)),          # -> (4, 10)
        ("2D Medium", (32, 250)),       # -> (32, 50)
        ("2D Large", (64, 1000)),       # -> (64, 200)
        ("3D Small", (4, 64, 125)),     # -> (4, 64, 25)
        ("3D Medium", (8, 128, 250)),   # -> (8, 128, 50)
        ("3D Large", (16, 256, 500)),   # -> (16, 256, 100)
        ("4D Small", (4, 15, 8, 8)),    # -> (4, 3, 8, 8)
        ("4D Medium", (8, 30, 16, 16)), # -> (8, 6, 16, 16)
        ("4D Large", (16, 60, 32, 32)), # -> (16, 12, 32, 32)
    ]
    
    print(f"\n{'Test Case':<30} | {'Time (ms)':<12} | Output Shape")
    print("-" * 70)
    
    results = {}
    for name, shape in test_cases:
        x = torch.randn(*shape, device=device)
        avg_time, output = benchmark_activation(activation, x, name)
        results[name] = avg_time
    
    print("=" * 70)
    print("\nMemory efficiency test (4D Large):")
    x = torch.randn(16, 60, 32, 32, device=device)
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
        _ = activation(x)
        peak_mem = torch.cuda.max_memory_allocated() / 1024**2  # MB
        print(f"Peak GPU memory: {peak_mem:.2f} MB")
    
    print("\nGradient computation test:")
    x = torch.randn(8, 125, device=device, requires_grad=True)
    start = time.time()
    output = activation(x)
    loss = output.sum()
    loss.backward()
    elapsed = (time.time() - start) * 1000
    print(f"Forward + backward pass: {elapsed:.4f} ms")
    
    print("\n✓ All benchmarks completed successfully!")


if __name__ == "__main__":
    main()
