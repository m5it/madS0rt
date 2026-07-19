#!/usr/bin/env python3
"""
GPU Acceleration Examples for madS0rt.

Demonstrates how to use GPU acceleration for large-scale numeric sorting.
Requires CuPy and CUDA-capable GPU.
"""

import time
import random
import sys

# Check GPU availability
try:
    from madsort import gpu_available, MadSorter, madsorted, gpu_sort
    if not gpu_available():
        print("WARNING: GPU not available. Install CuPy with CUDA support.")
        print("  pip install cupy-cuda11x  # For CUDA 11.x")
        print("  pip install cupy-cuda12x  # For CUDA 12.x")
        sys.exit(1)
except ImportError as e:
    print(f"ERROR: {e}")
    sys.exit(1)

print("=" * 70)
print("madS0rt GPU Acceleration Examples")
print("=" * 70)


def example_1_basic_gpu_sorting():
    """Example 1: Basic GPU sorting of large numeric arrays."""
    print("\nExample 1: Basic GPU Sorting")
    print("-" * 50)
    
    # Generate large dataset
    size = 1_000_000  # 1 million items
    print(f"Generating {size:,} random integers...")
    data = [random.randint(0, 1_000_000) for _ in range(size)]
    
    # CPU sort
    print("Sorting on CPU...")
    start = time.perf_counter()
    cpu_result = sorted(data)
    cpu_time = time.perf_counter() - start
    print(f"  CPU time: {cpu_time:.3f}s")
    
    # GPU sort
    print("Sorting on GPU...")
    start = time.perf_counter()
    gpu_result = gpu_sort(data, min_size=10000)
    gpu_time = time.perf_counter() - start
    print(f"  GPU time: {gpu_time:.3f}s")
    
    # Verify correctness
    assert cpu_result == gpu_result, "Results don't match!"
    
    speedup = cpu_time / gpu_time
    print(f"\n  Speedup: {speedup:.2f}x faster on GPU")
    
    return cpu_time, gpu_time


def example_2_hybrid_sorting():
    """Example 2: Hybrid CPU/GPU sorting with mixed data sizes."""
    print("\nExample 2: Hybrid CPU/GPU Sorting")
    print("-" * 50)
    
    # Create mixed dataset: some small buckets, some large
    print("Creating mixed dataset...")
    
    # Many small buckets (will use CPU)
    small_data = []
    for i in range(100):  # 100 small buckets
        small_data.extend([f"bucket_{i}_{j}" for j in range(50)])
    
    # Few large buckets (will use GPU)
    large_numeric = [random.random() for _ in range(500_000)]
    
    # Combine
    mixed_data = small_data + large_numeric
    
    print(f"Total items: {len(mixed_data):,}")
    print(f"  Small buckets: {len(small_data):,} (CPU)")
    print(f"  Large numeric: {len(large_numeric):,} (GPU)")
    
    # Sort with GPU enabled
    print("\nSorting with GPU acceleration...")
    sorter = MadSorter(
        prefix_length=2,
        use_gpu=True,
        gpu_threshold=10000
    )
    
    start = time.perf_counter()
    result = sorter.sort(mixed_data.copy())
    elapsed = time.perf_counter() - start
    
    print(f"  Total time: {elapsed:.3f}s")
    
    # Check stats
    stats = sorter.get_stats()
    print(f"\nSorting statistics:")
    print(f"  Total buckets: {stats['num_buckets']}")
    print(f"  GPU buckets sorted: {stats.get('gpu_buckets_sorted', 0)}")
    print(f"  CPU buckets sorted: {stats.get('cpu_buckets_sorted', 0)}")
    
    return elapsed


def example_3_performance_comparison():
    """Example 3: Detailed performance comparison across sizes."""
    print("\nExample 3: Performance Comparison")
    print("-" * 50)
    
    sizes = [10_000, 100_000, 500_000, 1_000_000]
    
    print(f"{'Size':>12} {'CPU (s)':>10} {'GPU (s)':>10} {'Speedup':>10}")
    print("-" * 50)
    
    for size in sizes:
        # Generate data
        data = [random.random() for _ in range(size)]
        
        # CPU sort
        start = time.perf_counter()
        cpu_result = sorted(data)
        cpu_time = time.perf_counter() - start
        
        # GPU sort
        start = time.perf_counter()
        gpu_result = gpu_sort(data, min_size=1000)
        gpu_time = time.perf_counter() - start
        
        # Verify
        assert cpu_result == pytest.approx(gpu_result, rel=1e-9), f"Mismatch at size {size}"
        
        speedup = cpu_time / gpu_time
        print(f"{size:>12,} {cpu_time:>10.3f} {gpu_time:>10.3f} {speedup:>9.2f}x")


def example_4_gpu_threshold_tuning():
    """Example 4: Tuning GPU threshold for optimal performance."""
    print("\nExample 4: GPU Threshold Tuning")
    print("-" * 50)
    
    # Generate dataset
    size = 500_000
    print(f"Dataset size: {size:,} items")
    data = [random.random() for _ in range(size)]
    
    thresholds = [1000, 5000, 10000, 50000, 100000]
    
    print(f"\n{'Threshold':>12} {'Time (s)':>10} {'GPU Buckets':>12}")
    print("-" * 40)
    
    for threshold in thresholds:
        sorter = MadSorter(
            use_gpu=True,
            gpu_threshold=threshold
        )
        
        start = time.perf_counter()
        result = sorter.sort(data.copy())
        elapsed = time.perf_counter() - start
        
        stats = sorter.get_stats()
        gpu_buckets = stats.get('gpu_buckets_sorted', 0)
        
        print(f"{threshold:>12,} {elapsed:>10.3f} {gpu_buckets:>12}")
    
    print("\nOptimal threshold depends on data distribution and GPU model.")


def example_5_memory_limitations():
    """Example 5: Handling GPU memory limitations."""
    print("\nExample 5: GPU Memory Management")
    print("-" * 50)
    
    from madsort.gpu_backend import (
        get_gpu_memory_info,
        estimate_gpu_memory_needed,
        can_fit_in_gpu_memory
    )
    
    # Check available memory
    mem_info = get_gpu_memory_info()
    if mem_info:
        print(f"GPU Memory:")
        print(f"  Total: {mem_info['total'] / 1e9:.2f} GB")
        print(f"  Used:  {mem_info['used'] / 1e9:.2f} GB")
        print(f"  Free:  {mem_info['free'] / 1e9:.2f} GB")
    
    # Test different sizes
    sizes = [100_000, 1_000_000, 10_000_000, 100_000_000]
    
    print(f"\n{'Size':>12} {'Memory Needed':>15} {'Can Fit':>10}")
    print("-" * 45)
    
    for size in sizes:
        needed = estimate_gpu_memory_needed(size, item_size_bytes=8)
        can_fit = can_fit_in_gpu_memory(size, item_size_bytes=8)
        
        print(f"{size:>12,} {needed/1e9:>14.2f} GB {str(can_fit):>10}")
    
    # Demonstrate automatic fallback
    print("\nDemonstrating automatic CPU fallback...")
    print("  (Creating data that might not fit - will fallback to CPU)")
    
    # Create data
    data = [random.random() for _ in range(1_000_000)]
    
    sorter = MadSorter(use_gpu=True, gpu_threshold=1000)
    
    start = time.perf_counter()
    result = sorter.sort(data)
    elapsed = time.perf_counter() - start
    
    print(f"  Sorted in {elapsed:.3f}s (automatic fallback if needed)")


def example_6_real_world_numeric_data():
    """Example 6: Real-world numeric data processing."""
    print("\nExample 6: Real-World Numeric Data")
    print("-" * 50)
    
    # Simulate scientific data: sensor readings
    print("Simulating sensor readings...")
    num_sensors = 1000
    readings_per_sensor = 1000
    
    # Generate sensor data
    sensor_data = []
    for sensor_id in range(num_sensors):
        for reading_id in range(readings_per_sensor):
            # Simulate temperature reading with noise
            temp = 20.0 + random.gauss(0, 5) + sensor_id * 0.01
            sensor_data.append({
                'sensor_id': sensor_id,
                'reading_id': reading_id,
                'temperature': temp,
                'timestamp': reading_id
            })
    
    print(f"Total readings: {len(sensor_data):,}")
    
    # Sort by temperature using GPU
    print("\nSorting by temperature (GPU)...")
    sorter = MadSorter(
        key_func=lambda x: x['temperature'],
        use_gpu=True,
        gpu_threshold=50000
    )
    
    start = time.perf_counter()
    sorted_data = sorter.sort(sensor_data)
    elapsed = time.perf_counter() - start
    
    print(f"  Sorted in {elapsed:.3f}s")
    
    # Show extremes
    print(f"\nTemperature statistics:")
    print(f"  Lowest:  {sorted_data[0]['temperature']:.2f}°C")
    print(f"  Highest: {sorted_data[-1]['temperature']:.2f}°C")
    
    # Verify sorted
    for i in range(len(sorted_data) - 1):
        assert sorted_data[i]['temperature'] <= sorted_data[i+1]['temperature']
    
    return elapsed


def example_7_mixed_types_with_gpu():
    """Example 7: Mixed types - GPU for numeric, CPU for strings."""
    print("\nExample 7: Mixed Data Types")
    print("-" * 50)
    
    # Create dataset with both strings and numbers
    mixed_data = {
        'users': [f"user_{i}" for i in range(10000)],
        'scores': [random.randint(0, 1000) for _ in range(100000)],
        'prices': [random.random() * 1000 for _ in range(50000)],
    }
    
    print("Dataset composition:")
    for key, value in mixed_data.items():
        print(f"  {key}: {len(value):,} items")
    
    # Sort each with GPU where appropriate
    print("\nSorting with GPU acceleration...")
    
    results = {}
    for key, data in mixed_data.items():
        is_numeric = isinstance(data[0], (int, float))
        
        sorter = MadSorter(
            use_gpu=is_numeric,  # Only use GPU for numeric
            gpu_threshold=10000
        )
        
        start = time.perf_counter()
        results[key] = sorter.sort(data.copy())
        elapsed = time.perf_counter() - start
        
        gpu_used = sorter.get_stats().get('gpu_buckets_sorted', 0) > 0
        print(f"  {key}: {elapsed:.3f}s (GPU: {gpu_used})")
    
    return results


def main():
    """Run all examples."""
    print("\nRunning GPU acceleration examples...\n")
    
    try:
        # Run examples
        example_1_basic_gpu_sorting()
        example_2_hybrid_sorting()
        example_3_performance_comparison()
        example_4_gpu_threshold_tuning()
        example_5_memory_limitations()
        example_6_real_world_numeric_data()
        example_7_mixed_types_with_gpu()
        
        print("\n" + "=" * 70)
        print("All examples completed successfully!")
        print("=" * 70)
        
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    # Import pytest for approx comparison
    try:
        import pytest
    except ImportError:
        # Simple approx for verification
        class DummyPytest:
            @staticmethod
            def approx(a, rel=1e-9):
                return a
        pytest = DummyPytest()
    
    main()
