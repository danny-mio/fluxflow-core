# Paperspace CUDA Testing Instructions

## Branch Information
- **Branch:** `feature/bezier-performance-optimization`
- **GitHub URL:** https://github.com/danny-mio/fluxflow-core/tree/feature/bezier-performance-optimization

## Setup Commands (Run Once)

```bash
# Clone the repository (if not already cloned)
git clone https://github.com/danny-mio/fluxflow-core.git
cd fluxflow-core

# Checkout the feature branch
git fetch origin
git checkout feature/bezier-performance-optimization

# Create a virtual environment
python3 -m venv venv
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install the package in development mode with test dependencies
pip install -e ".[test]"

# Verify CUDA is available
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda}'); print(f'Device count: {torch.cuda.device_count()}')"
```

## Test Commands

### 1. Run Full Test Suite with CUDA

```bash
# Run all tests (should take ~5-10 minutes)
pytest tests/ -v --tb=short

# Save output to file for review
pytest tests/ -v --tb=short > test_output_cuda.txt 2>&1
```

### 2. Run Optimization-Specific Tests

```bash
# Test only the new optimization features
pytest tests/unit/test_optimizations.py -v

# Save output
pytest tests/unit/test_optimizations.py -v > optimization_tests_cuda.txt 2>&1
```

### 3. Run Performance Benchmarks

```bash
# Run the comprehensive benchmark suite (CRITICAL for validating performance)
python scripts/benchmark_optimizations.py

# Save benchmark results
python scripts/benchmark_optimizations.py > benchmark_cuda_results.txt 2>&1
```

### 4. Run Original Benchmark Script

```bash
# Run the existing benchmark on CUDA
python benchmark_bezier.py

# Save results
python benchmark_bezier.py > bezier_benchmark_cuda.txt 2>&1
```

### 5. Quick Smoke Test (If Short on Time)

```bash
# Minimal test to verify CUDA functionality
python -c "
import torch
from fluxflow.models.activations import BezierActivation

# Test on CUDA
device = 'cuda'
print(f'Testing on {device}')

activation = BezierActivation('sigmoid', None).to(device)
x = torch.randn(16, 60, 32, 32, device=device)

# Forward pass
output = activation(x)
print(f'Input shape: {x.shape}')
print(f'Output shape: {output.shape}')
print(f'Output device: {output.device}')
print(f'No NaNs: {not torch.isnan(output).any()}')

# Backward pass
output.sum().backward()
print('Backward pass successful')

print('✓ CUDA smoke test passed!')
"
```

### 6. Memory and Performance Profiling (Optional)

```bash
# Profile memory usage
python -c "
import torch
from fluxflow.models.activations import BezierActivation

device = 'cuda'
activation = BezierActivation('sigmoid', None).to(device)
x = torch.randn(16, 60, 32, 32, device=device)

torch.cuda.reset_peak_memory_stats()
output = activation(x)
peak_mem_mb = torch.cuda.max_memory_allocated() / 1024**2
print(f'Peak GPU memory: {peak_mem_mb:.2f} MB')
"
```

## Expected Results

### Test Suite
- **Expected:** All 429 tests should pass
- **Known issues:** None (all tests passed on CPU/MPS)

### Benchmark Performance
- **JIT Coverage:** Should report 100% (25/25 variants)
- **Power Cache:** Should show 100% hit rate after warmup
- **Expected speedup:** 15-35% vs baseline (varies by GPU)

### Memory Usage
- **4D Large test (16×60×32×32):** ~50-100 MB peak GPU memory
- **No memory leaks:** Memory should be stable across iterations

## What to Send Back

Please send the following files/outputs:

1. **Test results:** `test_output_cuda.txt` or console output from pytest
2. **Benchmark results:** `benchmark_cuda_results.txt` or console output
3. **GPU info:** Output from the CUDA verification command
4. **Any errors:** Full error messages if any tests fail

## Troubleshooting

### If CUDA is not available:
```bash
# Check CUDA installation
nvcc --version
nvidia-smi

# Verify PyTorch CUDA support
python -c "import torch; print(torch.__version__); print(torch.version.cuda)"
```

### If tests fail:
- Check the error message for specific test failures
- Run individual failing tests with: `pytest tests/path/to/test.py::TestClass::test_method -v`
- Send the full traceback

### If memory errors occur:
- Reduce batch sizes in benchmarks
- Check GPU memory with `nvidia-smi`
- Clear cache with `torch.cuda.empty_cache()`

## Quick Command Summary

```bash
# Full test + benchmark (recommended)
cd fluxflow-core
git checkout feature/bezier-performance-optimization
source venv/bin/activate
pytest tests/unit/test_optimizations.py -v > optimization_tests.txt 2>&1
python scripts/benchmark_optimizations.py > benchmark_results.txt 2>&1
```

## Questions?

If you encounter any issues or have questions, send me:
1. The command you ran
2. The full error message
3. Output from `nvidia-smi` and `python -c "import torch; print(torch.cuda.is_available())"`
