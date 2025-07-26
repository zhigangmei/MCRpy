"""
   Copyright 2025 MCRpy Multi-GPU Support Extension

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
"""
from __future__ import annotations

import os
import logging
from typing import Optional, List
import tensorflow as tf


class GPUManager:
    """Manages GPU configuration and distribution strategies for MCRpy."""

    def __init__(self):
        self.strategy = None
        self.gpus = None
        self.num_gpus = 0
        self.memory_growth_enabled = False

    def configure_gpus(self,
                       gpu_ids: Optional[List[int]] = None,
                       memory_growth: bool = True,
                       memory_limit: Optional[int] = None,
                       enable_mixed_precision: bool = False) -> None:
        """
        Configure GPU settings for MCRpy.

        Args:
            gpu_ids: List of GPU IDs to use. If None, use all available GPUs.
            memory_growth: Whether to enable memory growth (prevents TF from
                allocating all GPU memory).
            memory_limit: Memory limit in MB per GPU. If None, no limit is set.
            enable_mixed_precision: Whether to enable mixed precision training.
        """
        try:
            # Get list of physical GPUs
            physical_gpus = tf.config.list_physical_devices('GPU')
            
            if not physical_gpus:
                logging.warning("No GPUs detected. Running on CPU.")
                # Default strategy (no distribution)
                self.strategy = tf.distribute.get_strategy()
                return

            gpu_info = f"Detected {len(physical_gpus)} GPU(s): {physical_gpus}"
            logging.info(gpu_info)

            # Filter GPUs based on gpu_ids if provided
            if gpu_ids is not None:
                if max(gpu_ids) >= len(physical_gpus):
                    error_msg = (f"GPU ID {max(gpu_ids)} not available. "
                                 f"Only {len(physical_gpus)} GPUs detected.")
                    raise ValueError(error_msg)
                selected_gpus = [physical_gpus[i] for i in gpu_ids]
            else:
                selected_gpus = physical_gpus

            self.gpus = selected_gpus
            self.num_gpus = len(selected_gpus)

            # Configure memory growth and limits
            for gpu in selected_gpus:
                if memory_growth and not self.memory_growth_enabled:
                    tf.config.experimental.set_memory_growth(gpu, True)
                    self.memory_growth_enabled = True

                if memory_limit is not None:
                    logical_config = tf.config.LogicalDeviceConfiguration(
                        memory_limit=memory_limit)
                    tf.config.set_logical_device_configuration(
                        gpu, [logical_config])

            # Set up mixed precision if requested
            if enable_mixed_precision:
                tf.keras.mixed_precision.set_global_policy('mixed_float16')
                logging.info("Mixed precision enabled (float16)")

            # Set visible devices
            tf.config.set_visible_devices(selected_gpus, 'GPU')

            # Create distribution strategy
            if self.num_gpus > 1:
                device_names = [gpu.name for gpu in selected_gpus]
                self.strategy = tf.distribute.MirroredStrategy(
                    devices=device_names)
                strategy_info = (f"Using MirroredStrategy with {self.num_gpus} "
                                 f"GPUs: {device_names}")
                logging.info(strategy_info)
            else:
                self.strategy = tf.distribute.get_strategy()
                logging.info(f"Using single GPU: {selected_gpus[0].name}")
                
        except Exception as e:
            logging.error(f"Failed to configure GPUs: {e}")
            logging.info("Falling back to CPU/default strategy")
            self.strategy = tf.distribute.get_strategy()
            
    def get_strategy(self) -> tf.distribute.Strategy:
        """Get the current distribution strategy."""
        if self.strategy is None:
            return tf.distribute.get_strategy()
        return self.strategy
        
    def is_distributed(self) -> bool:
        """Check if we're using a distributed strategy."""
        return self.num_gpus > 1
        
    def get_replica_batch_size(self, global_batch_size: int) -> int:
        """Calculate per-replica batch size for distributed training."""
        if self.is_distributed():
            return global_batch_size // self.num_gpus
        return global_batch_size
        
    def distribute_dataset(self, dataset):
        """Distribute a dataset across replicas."""
        if self.is_distributed() and self.strategy is not None:
            return self.strategy.experimental_distribute_dataset(dataset)
        return dataset
        
    def print_gpu_info(self):
        """Print information about current GPU configuration."""
        print("=" * 50)
        print("GPU Configuration:")
        gpu_count = len(tf.config.list_physical_devices('GPU'))
        print(f"  Available GPUs: {gpu_count}")
        print(f"  Used GPUs: {self.num_gpus}")
        if self.gpus:
            for i, gpu in enumerate(self.gpus):
                print(f"    GPU {i}: {gpu.name}")
        print(f"  Strategy: {type(self.strategy).__name__}")
        print(f"  Memory growth: {self.memory_growth_enabled}")
        print("=" * 50)


# Global GPU manager instance
gpu_manager = GPUManager()


def setup_gpus(gpu_ids: Optional[List[int]] = None,
               memory_growth: bool = True,
               memory_limit: Optional[int] = None,
               enable_mixed_precision: bool = False) -> tf.distribute.Strategy:
    """
    Convenient function to set up GPU configuration.
    
    Args:
        gpu_ids: List of GPU IDs to use. If None, use all available GPUs.
        memory_growth: Whether to enable memory growth.
        memory_limit: Memory limit in MB per GPU.
        enable_mixed_precision: Whether to enable mixed precision.
        
    Returns:
        The configured distribution strategy.
    """
    gpu_manager.configure_gpus(
        gpu_ids=gpu_ids,
        memory_growth=memory_growth,
        memory_limit=memory_limit,
        enable_mixed_precision=enable_mixed_precision
    )
    return gpu_manager.get_strategy()


def get_strategy() -> tf.distribute.Strategy:
    """Get the current distribution strategy."""
    return gpu_manager.get_strategy()


def is_distributed() -> bool:
    """Check if we're using distributed training."""
    return gpu_manager.is_distributed()


def print_gpu_status():
    """Print current GPU status."""
    gpu_manager.print_gpu_info()


def auto_configure_gpus():
    """Automatically configure GPUs with sensible defaults."""
    # Check for environment variables first
    gpu_ids_env = os.environ.get('CUDA_VISIBLE_DEVICES', None)
    
    if gpu_ids_env is not None:
        try:
            gpu_ids = [int(x) for x in gpu_ids_env.split(',') if x.strip()]
            logging.info(f"Using GPUs from CUDA_VISIBLE_DEVICES: {gpu_ids}")
        except ValueError:
            warning_msg = ("Invalid CUDA_VISIBLE_DEVICES format, "
                           "using all available GPUs")
            logging.warning(warning_msg)
            gpu_ids = None
    else:
        gpu_ids = None
        
    return setup_gpus(
        gpu_ids=gpu_ids,
        memory_growth=True,
        memory_limit=None,
        enable_mixed_precision=False
    )
