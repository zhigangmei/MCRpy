#!/usr/bin/env python3
"""
Example script demonstrating multi-GPU usage in MCRpy.

This script shows how to:
1. Configure multiple GPUs
2. Use distributed loss computation
3. Monitor GPU usage during reconstruction
"""

import os
import numpy as np
import tensorflow as tf

# Import MCRpy components
import mcrpy
from mcrpy.src.gpu_manager import setup_gpus, print_gpu_status, is_distributed


def example_multi_gpu_reconstruction():
    """Demonstrate multi-GPU reconstruction with a synthetic microstructure."""
    
    print("=== MCRpy Multi-GPU Example ===\n")
    
    # 1. Check available GPUs
    print("1. Checking GPU availability...")
    gpus = tf.config.list_physical_devices('GPU')
    print(f"   Found {len(gpus)} GPU(s): {[gpu.name for gpu in gpus]}")
    
    if len(gpus) == 0:
        print("   No GPUs found, running on CPU")
        use_multi_gpu = False
    else:
        use_multi_gpu = len(gpus) > 1
        print(f"   Multi-GPU available: {use_multi_gpu}")
    
    # 2. Configure GPU strategy
    print("\n2. Configuring GPU strategy...")
    if use_multi_gpu:
        # Use all available GPUs
        strategy = setup_gpus(
            gpu_ids=None,  # Use all GPUs
            memory_growth=True,
            memory_limit=None,
            enable_mixed_precision=False
        )
    else:
        # Single GPU or CPU
        strategy = setup_gpus(
            gpu_ids=[0] if len(gpus) > 0 else None,
            memory_growth=True,
            memory_limit=None,
            enable_mixed_precision=False
        )
    
    print_gpu_status()
    
    # 3. Create a synthetic 3D microstructure
    print("\n3. Creating synthetic microstructure...")
    np.random.seed(42)
    size = 64  # Small size for demonstration
    synthetic_ms = np.random.random((size, size, size)) > 0.5
    synthetic_ms = synthetic_ms.astype(np.float64)
    
    print(f"   Microstructure shape: {synthetic_ms.shape}")
    print(f"   Volume fraction: {np.mean(synthetic_ms):.3f}")
    
    # 4. Characterize the microstructure
    print("\n4. Characterizing microstructure...")
    characterization_settings = mcrpy.CharacterizationSettings(
        descriptor_types=['Correlations', 'Variation'],
        limit_to=8,  # Reduced for faster computation
        use_multigrid_descriptor=True
    )
    
    ms = mcrpy.Microstructure(synthetic_ms, use_multiphase=False)
    descriptor = mcrpy.characterize(ms, characterization_settings)
    
    print(f"   Generated {len(descriptor)} descriptor(s)")
    
    # 5. Set up reconstruction with multi-GPU support
    print("\n5. Setting up reconstruction...")
    reconstruction_shape = (size, size, size)
    
    reconstruction_settings = mcrpy.ReconstructionSettings(
        descriptor_types=['Correlations', 'Variation'],
        descriptor_weights=[1.0, 100.0],
        limit_to=8,
        max_iter=50,  # Reduced for demonstration
        use_multigrid_reconstruction=use_multi_gpu,
        use_multigrid_descriptor=True,
        optimizer_type='Adam',  # Use gradient-based optimizer for GPU
        learning_rate=0.01,
        # Multi-GPU specific settings
        use_distributed_loss=use_multi_gpu and is_distributed(),
        greedy=True,
        batch_size=4 if use_multi_gpu else 1
    )
    
    print(f"   Target shape: {reconstruction_shape}")
    print(f"   Using distributed loss: {reconstruction_settings.use_distributed_loss}")
    print(f"   Using multigrid: {reconstruction_settings.use_multigrid_reconstruction}")
    
    # 6. Run reconstruction
    print("\n6. Running reconstruction...")
    print("   This may take a few minutes...")
    
    # Create DMCR with GPU configuration
    dmcr = mcrpy.DMCR(
        **vars(reconstruction_settings),
        # Additional GPU settings
        gpu_ids=None,  # Use configured GPUs
        memory_growth=True,
        use_distributed_loss=use_multi_gpu and is_distributed()
    )
    
    try:
        convergence_data, reconstructed_ms = dmcr.reconstruct(
            descriptor, reconstruction_shape
        )
        
        print("   Reconstruction completed successfully!")
        
        # 7. Analyze results
        print("\n7. Analyzing results...")
        original_vf = np.mean(synthetic_ms)
        reconstructed_vf = np.mean(reconstructed_ms.to_numpy())
        
        print(f"   Original volume fraction: {original_vf:.3f}")
        print(f"   Reconstructed volume fraction: {reconstructed_vf:.3f}")
        print(f"   Volume fraction error: {abs(original_vf - reconstructed_vf):.3f}")
        print(f"   Final loss: {convergence_data['line_data'][-1]:.6f}")
        print(f"   Total iterations: {len(convergence_data['line_data'])}")
        
        # 8. Save results (optional)
        print("\n8. Saving results...")
        os.makedirs('multi_gpu_example_results', exist_ok=True)
        
        # Save reconstructed microstructure
        reconstructed_ms.to_npy('multi_gpu_example_results/reconstructed_ms.npy')
        
        # Save original for comparison
        np.save('multi_gpu_example_results/original_ms.npy', synthetic_ms)
        
        print("   Results saved to 'multi_gpu_example_results/'")
        
    except Exception as e:
        print(f"   Reconstruction failed: {e}")
        return False
    
    print("\n=== Example completed successfully! ===")
    return True


def example_gpu_configuration():
    """Demonstrate different GPU configuration options."""
    
    print("=== GPU Configuration Examples ===\n")
    
    # Example 1: Automatic configuration
    print("1. Automatic GPU configuration:")
    from mcrpy.src.gpu_manager import auto_configure_gpus
    auto_configure_gpus()
    print_gpu_status()
    
    # Example 2: Manual configuration
    print("\n2. Manual GPU configuration:")
    gpus = tf.config.list_physical_devices('GPU')
    
    if len(gpus) > 0:
        # Use first GPU only with memory limit
        strategy = setup_gpus(
            gpu_ids=[0],
            memory_growth=True,
            memory_limit=2048,  # 2GB limit
            enable_mixed_precision=False
        )
        print_gpu_status()
    
    # Example 3: Check distribution strategy
    print("\n3. Distribution strategy information:")
    from mcrpy.src.gpu_manager import get_strategy, is_distributed
    
    strategy = get_strategy()
    print(f"   Strategy type: {type(strategy).__name__}")
    print(f"   Is distributed: {is_distributed()}")
    print(f"   Number of replicas: {strategy.num_replicas_in_sync}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='MCRpy Multi-GPU Example')
    parser.add_argument('--example', choices=['reconstruction', 'configuration', 'both'],
                       default='both', help='Which example to run')
    parser.add_argument('--gpu_ids', nargs='+', type=int,
                       help='Specific GPU IDs to use')
    parser.add_argument('--memory_limit', type=int,
                       help='Memory limit per GPU in MB')
    
    args = parser.parse_args()
    
    # Configure GPUs if specified
    if args.gpu_ids is not None or args.memory_limit is not None:
        print("Setting up custom GPU configuration...")
        setup_gpus(
            gpu_ids=args.gpu_ids,
            memory_growth=True,
            memory_limit=args.memory_limit,
            enable_mixed_precision=False
        )
    
    # Run selected examples
    if args.example in ['configuration', 'both']:
        example_gpu_configuration()
    
    if args.example in ['reconstruction', 'both']:
        if args.example == 'both':
            print("\n" + "="*60 + "\n")
        success = example_multi_gpu_reconstruction()
        
        if success:
            print("\nTip: Try running with different GPU configurations:")
            print("  python multi_gpu_example.py --gpu_ids 0")
            print("  python multi_gpu_example.py --memory_limit 4096")
            print("  python multi_gpu_example.py --example configuration")
        else:
            print("\nExample failed. Check GPU setup and try again.")
