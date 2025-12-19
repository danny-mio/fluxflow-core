#!/usr/bin/env python3
"""
Memory profiling script for FluxTransformerBlock with Pillar technique.

This script measures actual memory allocations during forward pass to verify
documentation claims about "5× + pillars" memory usage.
"""

import torch
import torch.nn as nn
from fluxflow.models.flow import FluxTransformerBlock
from fluxflow.models.activations import BezierActivation


def bytes_to_mb(bytes_val):
    """Convert bytes to megabytes."""
    return bytes_val / (1024 * 1024)


def measure_tensor_memory(B, T_img, T_txt, D):
    """
    Measure memory for individual tensor allocations.
    
    Args:
        B: Batch size
        T_img: Image sequence length
        T_txt: Text sequence length
        D: Model dimension (d_model)
    
    Returns:
        dict: Memory breakdown in MB
    """
    # Each float32 tensor element = 4 bytes
    bytes_per_element = 4
    
    memory = {
        "img_seq": B * T_img * D * bytes_per_element,
        "text_seq": B * T_txt * D * bytes_per_element,
        "img_p0": B * T_img * D * bytes_per_element,
        "img_p1": B * T_img * D * bytes_per_element,
        "img_p2": B * T_img * D * bytes_per_element,
        "img_p3": B * T_img * D * bytes_per_element,
        "concatenated_5D": B * T_img * (5 * D) * bytes_per_element,
    }
    
    # Convert to MB
    memory_mb = {k: bytes_to_mb(v) for k, v in memory.items()}
    
    # Peak memory: Before concatenation, we have img_seq + 4 pillars
    # During concatenation, we temporarily have img_seq + 4 pillars + concatenated tensor
    peak_before_concat = (
        memory_mb["img_seq"] + 
        memory_mb["img_p0"] + 
        memory_mb["img_p1"] + 
        memory_mb["img_p2"] + 
        memory_mb["img_p3"]
    )
    
    peak_during_concat = peak_before_concat + memory_mb["concatenated_5D"]
    
    memory_mb["peak_before_concat"] = peak_before_concat
    memory_mb["peak_during_concat"] = peak_during_concat
    
    return memory_mb


def measure_actual_memory():
    """
    Measure actual PyTorch memory allocations during forward pass.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print()
    
    # Test parameters
    B = 2
    T_img = 256
    T_txt = 77
    D = 128
    n_head = 8
    
    print(f"Test Configuration:")
    print(f"  Batch size (B): {B}")
    print(f"  Image seq length (T_img): {T_img}")
    print(f"  Text seq length (T_txt): {T_txt}")
    print(f"  Model dimension (D): {D}")
    print(f"  Number of heads: {n_head}")
    print()
    
    # Calculate theoretical memory
    theoretical_mem = measure_tensor_memory(B, T_img, T_txt, D)
    print("=" * 70)
    print("THEORETICAL MEMORY ANALYSIS")
    print("=" * 70)
    print(f"Individual tensor sizes:")
    print(f"  img_seq [B={B}, T_img={T_img}, D={D}]: {theoretical_mem['img_seq']:.2f} MB")
    print(f"  text_seq [B={B}, T_txt={T_txt}, D={D}]: {theoretical_mem['text_seq']:.2f} MB")
    print(f"  Each pillar output [B={B}, T_img={T_img}, D={D}]: {theoretical_mem['img_p0']:.2f} MB")
    print(f"  Concatenated tensor [B={B}, T_img={T_img}, D={5*D}]: {theoretical_mem['concatenated_5D']:.2f} MB")
    print()
    print(f"Memory peaks:")
    print(f"  Before concatenation (img_seq + 4 pillars): {theoretical_mem['peak_before_concat']:.2f} MB")
    print(f"  During concatenation (+ 5D tensor): {theoretical_mem['peak_during_concat']:.2f} MB")
    print()
    
    # Count pillar parameters
    pillar_params_per = 3 * (D * D + D)  # 3 layers, each D×D + D bias
    total_pillar_params = 4 * pillar_params_per
    pillar_param_memory = total_pillar_params * 4 / (1024 * 1024)  # float32 = 4 bytes
    
    print(f"Pillar Parameters:")
    print(f"  Parameters per pillar (depth=3): {pillar_params_per:,}")
    print(f"  Total pillar parameters (4 pillars): {total_pillar_params:,}")
    print(f"  Parameter memory (float32): {pillar_param_memory:.2f} MB")
    print()
    
    # Actual measurement (focus on pillar + concatenation part)
    print("=" * 70)
    print("ACTUAL PILLAR + CONCATENATION MEASUREMENT")
    print("=" * 70)
    
    # Simulate the critical part of forward pass (lines 229-240 in flow.py)
    # This is where the memory peak occurs
    
    print("Simulating FluxTransformerBlock pillar processing...")
    print()
    
    # Create dummy img_seq (output from FFN layer before Bezier)
    img_seq = torch.randn(B, T_img, D, device=device)
    
    # Gating (line 230)
    g = torch.sigmoid(img_seq)
    
    # Create pillar layers
    from fluxflow.models.flow import pillarLayer
    p0_layer = pillarLayer(in_size=D, out_size=D, depth=3, activation=nn.SiLU()).to(device)
    p1_layer = pillarLayer(in_size=D, out_size=D, depth=3, activation=nn.SiLU()).to(device)
    p2_layer = pillarLayer(in_size=D, out_size=D, depth=3, activation=nn.SiLU()).to(device)
    p3_layer = pillarLayer(in_size=D, out_size=D, depth=3, activation=nn.SiLU()).to(device)
    
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
    
    with torch.no_grad():
        # Pillar outputs (lines 231-234)
        img_p0 = p0_layer(g)
        img_p1 = p1_layer(g)
        img_p2 = p2_layer(g)
        img_p3 = p3_layer(g)
        
        print(f"Created 4 pillar outputs, each shape: {list(img_p0.shape)}")
        
        # Concatenation (line 239) - this is the peak memory moment
        concatenated = torch.cat([img_seq, img_p0, img_p1, img_p2, img_p3], dim=-1)
        
        print(f"Concatenated tensor shape: {list(concatenated.shape)}")
        print()
        
        # BezierActivation processes it
        bezier_act = BezierActivation()
        output = bezier_act(concatenated)
        
        print(f"BezierActivation output shape: {list(output.shape)}")
    
    if device.type == "cuda":
        torch.cuda.synchronize()
        peak_memory = torch.cuda.max_memory_allocated()
        print(f"\nPeak GPU memory during pillar+concat: {bytes_to_mb(peak_memory):.2f} MB")
    else:
        print("\nCPU mode: Demonstrating shapes and flow (memory tracking limited)")
    
    print()
    print("=" * 70)
    print("MEMORY FORMULA VERIFICATION")
    print("=" * 70)
    print()
    print("Question: Is '5× + pillars' accurate?")
    print()
    print("Analysis:")
    print("  - 'pillar' is a training technique using 4 separate MLPs (p0, p1, p2, p3)")
    print("  - Each pillar outputs [B, T_img, D]")
    print("  - These outputs are concatenated with img_seq to form [B, T_img, 5D]")
    print("  - BezierActivation then processes this 5D tensor")
    print()
    print("Activation memory (during forward pass):")
    print(f"  - Before concat: img_seq + p0 + p1 + p2 + p3 = 5 × [B,T,D] = {theoretical_mem['peak_before_concat']:.2f} MB")
    print(f"  - During concat: 5 × [B,T,D] + 1 × [B,T,5D] = {theoretical_mem['peak_during_concat']:.2f} MB")
    print()
    print("Parameter memory (persistent):")
    print(f"  - 4 pillars: {pillar_param_memory:.2f} MB")
    print(f"  - BezierActivation (input-based): 0 MB (no learnable params)")
    print()
    print("Conclusion:")
    print("  The '5×' refers to activation memory (5 tensors of size [B,T,D])")
    print("  The '+ pillars' could refer to:")
    print("    (a) Parameter memory of pillar MLPs (persistent)")
    print("    (b) Concatenated 5D tensor (temporary during forward)")
    print()
    print("  This needs clarification in documentation:")
    print("    - Separate parameter memory from activation memory")
    print("    - Specify whether 'pillars' means MLP params or concat tensor")
    print("    - Note that concat tensor is temporary (freed after Bezier activation)")
    print()


if __name__ == "__main__":
    measure_actual_memory()
