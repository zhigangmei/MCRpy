"""
   Copyrightfrom typing import Union, Tuple, List

import tensorflow as tf
import numpy as np5 MCRpy Multi-GPU Support Extension

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
from typing import Union, Tuple, List

import tensorflow as tf
import numpy as np

from mcrpy.src.Microstructure import Microstructure
from mcrpy.src.gpu_manager import get_strategy, is_distributed


def make_2d_gradients_distributed(loss_function: callable) -> callable:
    """Make a distributed version of 2D gradient computation."""
    strategy = get_strategy()

    @tf.function
    def compute_loss_and_gradients(ms: Microstructure):
        with tf.GradientTape() as tape:
            loss = loss_function(ms)
        gradients = tape.gradient(loss, [ms.x])
        return loss, gradients

    if is_distributed():
        @tf.function
        def distributed_compute(ms: Microstructure):
            # Use strategy.run for distributed computation
            per_replica_losses, per_replica_grads = strategy.run(
                compute_loss_and_gradients, args=(ms,))
            # Reduce across replicas
            loss = strategy.reduce(tf.distribute.ReduceOp.MEAN,
                                   per_replica_losses, axis=None)
            gradients = [strategy.reduce(tf.distribute.ReduceOp.MEAN,
                                        grad, axis=None)
                        for grad in per_replica_grads]
            return loss, gradients
        return distributed_compute
    else:
        return compute_loss_and_gradients


def make_3d_gradients_distributed(
        loss_function: Union[callable, List[callable], Tuple[callable]],
        shape_3d: Tuple[int]) -> callable:
    """
    Distributed version of 3D gradient computation.
    Distributes slice processing across multiple GPUs.
    """
    strategy = get_strategy()
    anisotropic = isinstance(loss_function, (tuple, list))

    zero_grads = tf.constant(np.zeros((1, *shape_3d), dtype=np.float64),
                             dtype=tf.float64)
    grads = tf.Variable(initial_value=zero_grads, trainable=False,
                        dtype=tf.float64)

    if not is_distributed():
        # Fall back to original implementation for single GPU
        from mcrpy.src.loss_computation import make_3d_gradients
        return make_3d_gradients(loss_function, shape_3d)

    def process_slice_batch(ms: Microstructure, spatial_dim: int,
                           slice_indices: tf.Tensor, use_loss: callable):
        """Process a batch of slices on one replica."""
        partial_loss = 0.0
        partial_grads = tf.zeros_like(ms.x)

        for i in tf.range(tf.shape(slice_indices)[0]):
            slice_index = slice_indices[i]
            with tf.GradientTape() as tape:
                ms_slice = ms.get_slice(spatial_dim, slice_index)
                slice_loss = use_loss(ms_slice)
            slice_grads = tape.gradient(slice_loss, [ms.x])[0]
            partial_loss += slice_loss
            partial_grads += slice_grads

        return partial_loss, partial_grads

    @tf.function
    def distributed_gradient_accumulation(ms: Microstructure):
        grads.assign(zero_grads)
        total_loss = 0.0

        for spatial_dim in range(3):
            use_loss = (loss_function[spatial_dim] if anisotropic
                       else loss_function)

            # Distribute slices across replicas
            num_slices = ms.spatial_shape[spatial_dim]
            num_replicas = strategy.num_replicas_in_sync

            # Create slice batches for each replica
            slices_per_replica = num_slices // num_replicas
            remainder = num_slices % num_replicas

            slice_batches = []
            start_idx = 0
            for i in range(num_replicas):
                batch_size = (slices_per_replica + 1 if i < remainder
                             else slices_per_replica)
                if batch_size > 0:
                    end_idx = start_idx + batch_size
                    slice_batch = tf.range(start_idx, end_idx, dtype=tf.int32)
                    slice_batches.append(slice_batch)
                    start_idx = end_idx

            # Pad slice_batches to ensure all replicas have work
            while len(slice_batches) < num_replicas:
                slice_batches.append(tf.constant([], dtype=tf.int32))

            # Create a distributed dataset
            slice_dataset = tf.data.Dataset.from_tensor_slices(slice_batches)
            dist_dataset = strategy.experimental_distribute_dataset(
                slice_dataset)

            # Process batches across replicas
            def step_fn(slice_batch):
                return process_slice_batch(ms, spatial_dim, slice_batch,
                                          use_loss)

            per_replica_results = strategy.run(step_fn, args=(dist_dataset,))

            # Aggregate results
            dim_loss = strategy.reduce(tf.distribute.ReduceOp.SUM,
                                      per_replica_results[0], axis=None)
            dim_grads = strategy.reduce(tf.distribute.ReduceOp.SUM,
                                       per_replica_results[1], axis=None)

            total_loss += dim_loss
            grads.assign_add(dim_grads)

        return total_loss, [grads]

    return distributed_gradient_accumulation


def make_3d_gradients_greedy_distributed(
        loss_function: Union[callable, List[callable], Tuple[callable]],
        shape_3d: Tuple[int],
        batch_size: int = 1) -> callable:
    """
    Distributed version of greedy 3D gradient computation.
    """
    strategy = get_strategy()
    anisotropic = isinstance(loss_function, (tuple, list))

    zero_grads = tf.constant(np.zeros((1, *shape_3d), dtype=np.float64),
                             dtype=tf.float64)
    grads = tf.Variable(initial_value=zero_grads, trainable=False,
                        dtype=tf.float64)

    if not is_distributed():
        # Fall back to original implementation
        from mcrpy.src.loss_computation import make_3d_gradients_greedy
        return make_3d_gradients_greedy(loss_function, shape_3d, batch_size)

    @tf.function
    def distributed_greedy_accumulation(ms: Microstructure):
        grads.assign(zero_grads)
        total_loss = 0.0

        for spatial_dim in range(3):
            use_loss = (loss_function[spatial_dim] if anisotropic
                       else loss_function)

            n_slices = (batch_size if batch_size >= 1
                       else int(ms.spatial_shape[spatial_dim] * batch_size))

            # Distribute slice processing across replicas
            slices_per_replica = max(1, n_slices // strategy.num_replicas_in_sync)

            def process_replica_slices():
                replica_loss = 0.0
                replica_grads = tf.zeros_like(ms.x)

                for _ in range(slices_per_replica):
                    slice_index = tf.random.uniform(
                        [], minval=0, maxval=ms.spatial_shape[spatial_dim],
                        dtype=tf.int32)

                    with tf.GradientTape() as tape:
                        ms_slice = ms.get_slice(spatial_dim, slice_index)
                        inner_loss = use_loss(ms_slice)

                    partial_grads = tape.gradient(inner_loss, [ms.x])[0]
                    replica_loss += inner_loss
                    replica_grads += partial_grads

                return replica_loss, replica_grads

            # Run on all replicas
            per_replica_results = strategy.run(process_replica_slices)

            # Aggregate results
            dim_loss = strategy.reduce(tf.distribute.ReduceOp.MEAN,
                                      per_replica_results[0], axis=None)
            dim_grads = strategy.reduce(tf.distribute.ReduceOp.MEAN,
                                       per_replica_results[1], axis=None)

            total_loss += dim_loss
            grads.assign_add(dim_grads)

        return total_loss, [grads]

    return distributed_greedy_accumulation


def make_call_loss_distributed(
        loss_function: Union[callable, List[callable], Tuple[callable]],
        ms: Microstructure,
        is_gradient_based: bool,
        sparse: bool = False,
        greedy: bool = False,
        batch_size: int = 1) -> callable:
    """
    Distributed version of make_call_loss that chooses the appropriate
    computation method based on the distribution strategy.
    """
    if not is_distributed():
        # Fall back to original implementation
        from mcrpy.src.loss_computation import make_call_loss
        return make_call_loss(loss_function, ms, is_gradient_based,
                             sparse, greedy, batch_size)

    ms_is_3d = ms.is_3D
    ms_is_2d = not ms_is_3d

    if ms_is_2d and is_gradient_based:
        return make_2d_gradients_distributed(loss_function)
    elif ms_is_3d and is_gradient_based:
        if greedy:
            return make_3d_gradients_greedy_distributed(
                loss_function, ms.shape, batch_size=batch_size)
        else:
            return make_3d_gradients_distributed(loss_function, ms.shape)
    else:
        # For non-gradient based methods, use original implementation
        from mcrpy.src.loss_computation import make_call_loss
        return make_call_loss(loss_function, ms, is_gradient_based,
                             sparse, greedy, batch_size)
