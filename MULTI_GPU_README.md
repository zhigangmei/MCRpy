# Multi-GPU Support for MCRpy

This document describes the multi-GPU capabilities added to MCRpy for accelerated microstructure reconstruction.

## Overview

MCRpy now supports distributed computation across multiple GPUs using TensorFlow's distribution strategies. This can significantly accelerate the reconstruction process, especially for large 3D microstructures.

## Features

- **Automatic GPU Detection**: Automatically detects and configures available GPUs
- **Flexible GPU Selection**: Choose specific GPUs to use via command line or environment variables
- **Memory Management**: Configurable memory growth and limits to prevent OOM errors
- **Mixed Precision**: Optional float16 mixed precision for improved performance
- **Distributed Loss Computation**: Parallel slice processing for 3D reconstructions
- **Fallback Support**: Gracefully falls back to single GPU or CPU if multi-GPU setup fails

## Usage

### Command Line Interface

#### Basic Multi-GPU Usage
```bash
# Use all available GPUs (default)
python match.py --microstructure_filename ms.npy --auto_gpu_config

# Use specific GPUs
python match.py --microstructure_filename ms.npy --gpu_ids 0 1 2

# Disable automatic GPU configuration
python match.py --microstructure_filename ms.npy --no_auto_gpu_config
```

#### Advanced GPU Configuration
```bash
# Enable mixed precision training
python match.py --microstructure_filename ms.npy --enable_mixed_precision

# Set memory limit per GPU (in MB)
python match.py --microstructure_filename ms.npy --memory_limit 4096

# Disable memory growth
python match.py --microstructure_filename ms.npy --no_memory_growth

# Use distributed loss computation
python match.py --microstructure_filename ms.npy --use_distributed_loss
```

### Environment Variables

You can also control GPU usage via environment variables:

```bash
# Use specific GPUs
export CUDA_VISIBLE_DEVICES=0,1,2
python match.py --microstructure_filename ms.npy

# Hide all GPUs (CPU only)
export CUDA_VISIBLE_DEVICES=""
python match.py --microstructure_filename ms.npy
```

### Python API

```python
import mcrpy
from mcrpy.src.gpu_manager import setup_gpus

# Configure GPUs manually
strategy = setup_gpus(
    gpu_ids=[0, 1],
    memory_growth=True,
    memory_limit=4096,
    enable_mixed_precision=False
)

# Create DMCR with multi-GPU support
dmcr = mcrpy.DMCR(
    descriptor_types=['Correlations', 'Variation'],
    descriptor_weights=[1.0, 10.0],
    gpu_ids=[0, 1],
    memory_growth=True,
    use_distributed_loss=True
)

# Proceed with normal reconstruction
convergence_data, ms = dmcr.reconstruct(descriptor, (128, 128, 128))
```

## Command Line Arguments

### Available for all entry points (match.py, reconstruct.py, characterize.py)

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--gpu_ids` | int list | None | Specific GPU IDs to use (e.g., `--gpu_ids 0 1 2`) |
| `--memory_growth` | flag | True | Enable GPU memory growth |
| `--no_memory_growth` | flag | - | Disable GPU memory growth |
| `--memory_limit` | int | None | Memory limit per GPU in MB |
| `--enable_mixed_precision` | flag | False | Enable mixed precision training (float16) |
| `--use_distributed_loss` | flag | False | Use distributed loss computation for multi-GPU |
| `--auto_gpu_config` | flag | True | Automatically configure GPUs with sensible defaults |

## Performance Considerations

### When to Use Multi-GPU

- **3D Reconstructions**: Most beneficial for large 3D microstructures (>128³ voxels)
- **Multiple Descriptors**: When using many descriptor types simultaneously
- **Long Optimizations**: For reconstructions requiring many iterations

### Memory Requirements

- Each GPU needs to hold the full microstructure and gradients
- Memory usage scales with microstructure size and number of descriptors
- Use `--memory_limit` to prevent OOM errors on systems with limited GPU memory

### Optimal Configuration

```bash
# For large 3D reconstructions with 2-4 GPUs
python match.py \
    --microstructure_filename large_ms.npy \
    --add_dimension 256 \
    --gpu_ids 0 1 2 3 \
    --memory_growth \
    --use_distributed_loss \
    --batch_size 4

# For mixed precision acceleration
python match.py \
    --microstructure_filename ms.npy \
    --enable_mixed_precision \
    --memory_growth
```

## Troubleshooting

### Common Issues

1. **CUDA OOM Errors**
   ```bash
   # Solution: Reduce memory usage
   python match.py --memory_limit 4096 --batch_size 1
   ```

2. **No GPUs Detected**
   ```bash
   # Check CUDA installation
   python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
   ```

3. **Multi-GPU Performance Issues**
   ```bash
   # Ensure distributed loss is enabled for 3D
   python match.py --use_distributed_loss --batch_size 4
   ```

### Debug Information

Enable verbose GPU information:

```python
from mcrpy.src.gpu_manager import print_gpu_status
print_gpu_status()
```

## System Requirements

- **TensorFlow**: ≥2.7.1 with CUDA support
- **NVIDIA GPUs**: Compatible with CUDA 11.0+
- **CUDA**: 11.0 or later
- **cuDNN**: Compatible version with CUDA installation

## Best Practices

1. **Start Small**: Test with smaller microstructures first
2. **Monitor Memory**: Use `nvidia-smi` to monitor GPU memory usage
3. **Batch Size**: Adjust `--batch_size` based on available memory
4. **Mixed Precision**: Enable for modern GPUs (Volta architecture or newer)
5. **Memory Growth**: Keep enabled to prevent allocation issues

## Example Configurations

### Single Workstation (4 GPUs)
```bash
python match.py \
    --microstructure_filename ms_256x256x256.npy \
    --descriptor_types Correlations Variation GramMatrices \
    --gpu_ids 0 1 2 3 \
    --memory_growth \
    --use_distributed_loss \
    --batch_size 8
```

### High-Memory Server (8 GPUs)
```bash
python match.py \
    --microstructure_filename large_ms.npy \
    --add_dimension 512 \
    --gpu_ids 0 1 2 3 4 5 6 7 \
    --memory_limit 8192 \
    --enable_mixed_precision \
    --use_distributed_loss \
    --batch_size 16
```

### Development/Testing (Single GPU)
```bash
python match.py \
    --microstructure_filename test_ms.npy \
    --gpu_ids 0 \
    --memory_growth \
    --enable_mixed_precision
```
