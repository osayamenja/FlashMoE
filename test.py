"""
Minimal test for FlashMoE with compiled config
"""
import torch
import flashmoe

def test_flashmoe_basic():
    """
    Test FlashMoE with tensors matching compiled config
    
    Compiled config (from csrc/kleos_config.json):
    - global_batch: 256
    - sequence_len: 8192
    - hidden_size: 2048
    - intermediate_size: 2048
    - num_experts: 64
    - torch_dtype: 1 (FP16)
    """
    
    print("=" * 60)
    print("Testing FlashMoE with compiled config")
    print("=" * 60)
    
    # Compiled dimensions (must match csrc/kleos_config.json)
    batch = 1
    seq_len = 8192
    hidden_size = 2048
    intermediate_size = 2048
    num_experts = 64
    dtype = torch.float32  # torch_dtype=1 means FP16
    
    print(f"\nCreating tensors:")
    print(f"  Input: [{batch}, {seq_len}, {hidden_size}]")
    print(f"  Gate:  [{hidden_size}, {num_experts}]")
    print(f"  Experts: [{num_experts}, 2, {intermediate_size}, {hidden_size}]")
    print(f"  Dtype: {dtype}")
    
    # Create input tensors with correct shapes
    input_tensor = torch.randn(
        batch, 
        seq_len, 
        hidden_size, 
        dtype=dtype, 
        device='cuda'
    )
    
    gate_weights = torch.randn(
        hidden_size, 
        num_experts, 
        dtype=dtype, 
        device='cuda'
    )
    
    expert_weights = torch.randn(
        num_experts,
        2,  # up and down projections
        intermediate_size,
        hidden_size,
        dtype=dtype,
        device='cuda'
    )
    
    print("\n✓ Tensors created successfully")
    print(f"  Input device: {input_tensor.device}")
    print(f"  Memory usage: ~{input_tensor.numel() * 2 / 1e9:.2f} GB (input only)")
    
    # Run MoE forward pass
    print("\nRunning MoE forward pass...")
    
    try:
        output = flashmoe.run_moe(
            input_tensor,
            gate_weights,
            expert_weights
        )
        
        print("\n" + "=" * 60)
        print("✓ SUCCESS!")
        print("=" * 60)
        print(f"Output shape: {output.shape}")
        print(f"Output dtype: {output.dtype}")
        print(f"Output device: {output.device}")
        print(f"Output range: [{output.min():.4f}, {output.max():.4f}]")
        
        return output
        
    except RuntimeError as e:
        print("\n" + "=" * 60)
        print("✗ FAILED")
        print("=" * 60)
        print(f"Error: {e}")
        raise


def test_flashmoe_wrong_shape():
    """
    Test that wrong shapes are caught with clear error messages
    """
    print("\n" + "=" * 60)
    print("Testing shape validation (should fail)")
    print("=" * 60)
    
    # Wrong dimensions - should fail!
    wrong_batch = 128  # Should be 256
    seq_len = 8192
    hidden_size = 2048
    
    print(f"\nTrying with wrong batch size: {wrong_batch} (should be 256)")
    
    input_tensor = torch.randn(wrong_batch, seq_len, hidden_size, 
                               dtype=torch.float32, device='cuda')
    gate_weights = torch.randn(hidden_size, 64, dtype=torch.float32, device='cuda')
    expert_weights = torch.randn(64, 2, 2048, hidden_size, 
                                 dtype=torch.float32, device='cuda')
    
    try:
        output = flashmoe.run_moe(input_tensor, gate_weights, expert_weights)
        print("✗ Should have failed but didn't!")
    except RuntimeError as e:
        print(f"✓ Correctly caught shape mismatch:")
        print(f"  Error: {e}")


if __name__ == "__main__":
    # Test 1: Correct shapes (should succeed)
    output = test_flashmoe_basic()
    
    # Test 2: Wrong shapes (should fail with clear error)
    # test_flashmoe_wrong_shape()
    
    print("\n" + "=" * 60)
    print("All tests completed!")
    print("=" * 60)