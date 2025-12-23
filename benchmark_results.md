# Bezier Activation Performance Benchmarks

## Summary

This report contains comprehensive benchmarks for all Bezier activation optimizations.

### Optimizations Tested

1. **JIT Compilation**: 25 pre-compiled activation combinations
2. **Cached Power Computations**: LRU-cached computation graphs
3. **Cross-device Support**: CPU, CUDA, MPS (where available)

## Results by Device

### CPU

| Test Case | Time (ms) | Speedup |
|-----------|-----------|---------|
| 2D_large | 0.0962 | 1.00x |
| 2D_medium | 0.0360 | 1.00x |
| 2D_small | 0.0293 | 1.00x |
| 3D_medium | 0.7943 | 1.00x |
| 3D_small | 0.0643 | 1.00x |
| 4D_large | 1.1938 | 1.00x |
| 4D_medium | 0.0704 | 1.00x |
| 4D_small | 0.0302 | 1.00x |
| jit_none_none | 0.0539 | 1.00x |
| jit_none_relu | 0.0621 | 1.00x |
| jit_none_sigmoid | 0.1078 | 1.00x |
| jit_none_silu | 0.1241 | 1.00x |
| jit_none_tanh | 0.3135 | 1.00x |
| jit_relu_none | 0.0688 | 1.00x |
| jit_relu_relu | 0.0782 | 1.00x |
| jit_relu_sigmoid | 0.1210 | 1.00x |
| jit_relu_silu | 0.1160 | 1.00x |
| jit_relu_tanh | 0.3034 | 1.00x |
| jit_sigmoid_none | 0.0653 | 1.00x |
| jit_sigmoid_relu | 0.0875 | 1.00x |
| jit_sigmoid_sigmoid | 0.1195 | 1.00x |
| jit_sigmoid_silu | 0.1273 | 1.00x |
| jit_sigmoid_tanh | 0.3229 | 1.00x |
| jit_silu_none | 0.0654 | 1.00x |
| jit_silu_relu | 0.0877 | 1.00x |
| jit_silu_sigmoid | 0.1372 | 1.00x |
| jit_silu_silu | 0.1424 | 1.00x |
| jit_silu_tanh | 0.3106 | 1.00x |
| jit_tanh_none | 0.1290 | 1.00x |
| jit_tanh_relu | 0.1391 | 1.00x |
| jit_tanh_sigmoid | 0.1992 | 1.00x |
| jit_tanh_silu | 0.1940 | 1.00x |
| jit_tanh_tanh | 0.3499 | 1.00x |
| power_cache | 0.7876 | 1.00x |

### MPS

| Test Case | Time (ms) | Speedup |
|-----------|-----------|---------|
| 2D_large | 0.0472 | 1.00x |
| 2D_medium | 0.0479 | 1.00x |
| 2D_small | 0.0477 | 1.00x |
| 3D_medium | 0.0564 | 1.00x |
| 3D_small | 0.0623 | 1.00x |
| 4D_large | 0.0502 | 1.00x |
| 4D_medium | 0.0485 | 1.00x |
| 4D_small | 0.0493 | 1.00x |
| jit_none_none | 0.0533 | 1.00x |
| jit_none_relu | 0.1585 | 1.00x |
| jit_none_sigmoid | 0.0600 | 1.00x |
| jit_none_silu | 0.3088 | 1.00x |
| jit_none_tanh | 0.0577 | 1.00x |
| jit_relu_none | 0.0770 | 1.00x |
| jit_relu_relu | 0.1710 | 1.00x |
| jit_relu_sigmoid | 0.0844 | 1.00x |
| jit_relu_silu | 0.2090 | 1.00x |
| jit_relu_tanh | 0.0809 | 1.00x |
| jit_sigmoid_none | 0.0494 | 1.00x |
| jit_sigmoid_relu | 0.1714 | 1.00x |
| jit_sigmoid_sigmoid | 0.0524 | 1.00x |
| jit_sigmoid_silu | 0.2033 | 1.00x |
| jit_sigmoid_tanh | 0.0530 | 1.00x |
| jit_silu_none | 0.0900 | 1.00x |
| jit_silu_relu | 0.2227 | 1.00x |
| jit_silu_sigmoid | 0.0946 | 1.00x |
| jit_silu_silu | 0.2354 | 1.00x |
| jit_silu_tanh | 0.1010 | 1.00x |
| jit_tanh_none | 0.0476 | 1.00x |
| jit_tanh_relu | 0.1567 | 1.00x |
| jit_tanh_sigmoid | 0.0580 | 1.00x |
| jit_tanh_silu | 0.1873 | 1.00x |
| jit_tanh_tanh | 0.0538 | 1.00x |
| power_cache | 0.1824 | 1.00x |

## JIT Activation Combinations

All 25 combinations of t_pre_activation × p_preactivation are benchmarked:

- **t_activations**: None, sigmoid, tanh, silu, relu
- **p_activations**: None, sigmoid, tanh, silu, relu

All combinations are JIT-compiled for maximum performance.
