"""
Minimal test for FlashMoE run_moe()
"""
import torch
import flashmoe
import subprocess

def test_run_moe():
    """Test single-GPU MoE forward pass"""
    
    # Minimal config (matching kleos_config.json structure)
    config = {
        "capacity_factor": 1,
        "drop_tokens": 1,
        "expert_top_k": 2,
        "global_batch": 8,          # Small batch for testing
        "is_training": 0,
        "hidden_act": 0,
        "hidden_size": 128,         # Small for quick test
        "intermediate_size": 256,
        "mini_batch": 1,
        "moe_frequency": 1,
        "num_experts": 8,           # Fewer experts for testing
        "num_layers": 1,
        "sequence_len": 64,         # Short sequence for testing
        "torch_dtype": 1,           # FP16
        "vocab_size": 32000
    }
    
    print("=" * 60)
    print("Testing FlashMoE run_moe()")
    print("=" * 60)
    print(f"Config: global_batch={config['global_batch']}, "
          f"seq_len={config['sequence_len']}, "
          f"hidden_size={config['hidden_size']}")
    print(f"Total tokens: {config['global_batch'] * config['sequence_len']}")
    print("=" * 60)
    
    try:
        # Call run_moe (uses nvshmrun launcher internally)
        result = flashmoe.run_moe(config, n_processes=1)
        
        print("\n✓ SUCCESS!")
        print(f"Result: {result}")
        
    except subprocess.CalledProcessError as e:
        print("\n✗ FAILED - Subprocess error")
        print("=" * 60)
        print("STDOUT:")
        print(e.stdout)
        print("\nSTDERR:")
        print(e.stderr)
        print("=" * 60)
        
    except Exception as e:
        print(f"\n✗ FAILED - {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()


def test_with_larger_config():
    """Test with your original config"""
    
    config = {
        "capacity_factor": 1,
        "drop_tokens": 1,
        "expert_top_k": 2,
        "global_batch": 256,
        "is_training": 0,
        "hidden_act": 0,
        "hidden_size": 2048,
        "intermediate_size": 2048,
        "mini_batch": 1,
        "moe_frequency": 1,
        "num_experts": 64,
        "num_layers": 1,
        "sequence_len": 8192,
        "torch_dtype": 1,
        "vocab_size": 32000
    }
    
    print("\n" + "=" * 60)
    print("Testing with larger config")
    print("=" * 60)
    
    try:
        result = flashmoe.run_moe(config, n_processes=1)
        print("✓ SUCCESS!")
        print(f"Result: {result}")
        
    except subprocess.CalledProcessError as e:
        print("✗ FAILED")
        print("STDOUT:", e.stdout)
        print("STDERR:", e.stderr)
        
    except Exception as e:
        print(f"✗ FAILED: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    # Run minimal test first
    test_run_moe()
    
    # Uncomment to test with your original larger config
    # test_with_larger_config()