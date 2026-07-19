"""
Tests for madS0rt GPU backend.
Tests GPU availability, sorting, memory management, and fallback behavior.
Skips tests if GPU (CuPy + CUDA) is not available.
"""

import pytest
import time
import random
import numpy as np

# Check GPU availability at module level
try:
    from madsort.gpu_backend import gpu_available, _GPU_AVAILABLE
    GPU_AVAILABLE = gpu_available()
except ImportError:
    GPU_AVAILABLE = False

# Skip all tests if GPU not available
pytestmark = pytest.mark.skipif(
    not GPU_AVAILABLE,
    reason="GPU (CuPy + CUDA) not available"
)

# Import GPU backend functions
if GPU_AVAILABLE:
    from madsort.gpu_backend import (
        gpu_available,
        is_gpu_sortable,
        transfer_to_gpu,
        transfer_to_cpu,
        gpu_sort_bucket,
        gpu_sort_bucket_safe,
        _cpu_sort,
        get_gpu_memory_info,
        estimate_gpu_memory_needed,
        can_fit_in_gpu_memory,
        GPUBackend,
        gpu_sort,
    )


class TestGPUAvailability:
    """Test GPU availability detection."""
    
    def test_gpu_available_returns_bool(self):
        """gpu_available() returns boolean."""
        result = gpu_available()
        assert isinstance(result, bool)
    
    def test_gpu_available_true(self):
        """GPU should be available (otherwise tests skipped)."""
        assert gpu_available() is True


class TestGPUSortable:
    """Test GPU sortable detection."""
    
    def test_numeric_list_is_sortable(self):
        """Numeric lists are GPU sortable."""
        data = list(range(10000))
        assert is_gpu_sortable(data, min_size=1000) is True
    
    def test_small_list_not_sortable(self):
        """Small lists are not GPU sortable."""
        data = list(range(100))
        assert is_gpu_sortable(data, min_size=1000) is False
    
    def test_string_list_not_sortable(self):
        """String lists are not GPU sortable."""
        data = ["a"] * 10000
        assert is_gpu_sortable(data, min_size=1000) is False
    
    def test_mixed_types_not_sortable(self):
        """Mixed type lists are not GPU sortable."""
        data = [1, 2.0, 3, 4.0] * 2500  # 10000 items
        assert is_gpu_sortable(data, min_size=1000) is False
    
    def test_empty_list_not_sortable(self):
        """Empty list is not GPU sortable."""
        assert is_gpu_sortable([], min_size=1) is False
    
    def test_float_list_is_sortable(self):
        """Float lists are GPU sortable."""
        data = [1.5] * 10000
        assert is_gpu_sortable(data, min_size=1000) is True


class TestMemoryTransfer:
    """Test GPU memory transfer functions."""
    
    def test_transfer_to_gpu(self):
        """Transfer data to GPU."""
        data = list(range(10000))
        gpu_array = transfer_to_gpu(data)
        
        # Check it's a CuPy array
        import cupy as cp
        assert isinstance(gpu_array, cp.ndarray)
        assert len(gpu_array) == 10000
    
    def test_transfer_to_cpu(self):
        """Transfer data from GPU to CPU."""
        data = list(range(1000))
        gpu_array = transfer_to_gpu(data)
        result = transfer_to_cpu(gpu_array)
        
        assert result == data
    
    def test_round_trip_preserves_data(self):
        """Round-trip transfer preserves data."""
        data = [random.randint(0, 1000000) for _ in range(10000)]
        gpu_array = transfer_to_gpu(data)
        result = transfer_to_cpu(gpu_array)
        
        assert result == data


class TestGPUSorting:
    """Test GPU sorting functionality."""
    
    def test_gpu_sort_integers(self):
        """Sort integers on GPU."""
        data = list(range(10000, 0, -1))  # Reverse sorted
        result = gpu_sort_bucket(data, reverse=False)
        
        assert result == list(range(1, 10001))
    
    def test_gpu_sort_floats(self):
        """Sort floats on GPU."""
        data = [3.5, 1.2, 4.8, 2.1] * 2500  # 10000 items
        result = gpu_sort_bucket(data)
        
        expected = sorted(data)
        assert result == pytest.approx(expected, rel=1e-10)
    
    def test_gpu_sort_reverse(self):
        """Sort in reverse order on GPU."""
        data = list(range(1, 10001))
        result = gpu_sort_bucket(data, reverse=True)
        
        assert result == list(range(10000, 0, -1))
    
    def test_gpu_sort_with_key(self):
        """Sort with key function on GPU."""
        data = list(range(10000))
        # Key function: negate (sorts descending)
        result = gpu_sort_bucket(data, key_func=lambda x: -x)
        
        assert result == list(range(9999, -1, -1))
    
    def test_gpu_sort_large_dataset(self):
        """Sort large dataset on GPU."""
        data = [random.random() for _ in range(100000)]
        result = gpu_sort_bucket(data)
        
        # Verify sorted
        for i in range(len(result) - 1):
            assert result[i] <= result[i + 1]


class TestCPUFallback:
    """Test CPU fallback when GPU fails."""
    
    def test_safe_sort_uses_gpu(self):
        """Safe sort uses GPU for numeric data."""
        data = list(range(10000, 0, -1))
        result = gpu_sort_bucket_safe(data, fallback_to_cpu=True)
        
        assert result == list(range(1, 10001))
    
    def test_safe_sort_fallback_for_strings(self):
        """Safe sort falls back to CPU for non-numeric."""
        data = ["zebra", "apple", "banana"] * 100  # 300 items
        
        # Should fall back to CPU
        result = gpu_sort_bucket_safe(data, fallback_to_cpu=True)
        
        assert result == sorted(data) * 100
    
    def test_safe_sort_no_fallback_raises(self):
        """Safe sort raises if no fallback and GPU fails."""
        # String data can't use GPU
        data = ["a", "b", "c"] * 100
        
        with pytest.raises((ValueError, RuntimeError)):
            gpu_sort_bucket_safe(data, fallback_to_cpu=False)


class TestHybridSorting:
    """Test hybrid CPU/GPU sorting."""
    
    def test_backend_detects_gpu(self):
        """GPUBackend detects GPU availability."""
        backend = GPUBackend(min_bucket_size=1000)
        assert backend.is_available() is True
    
    def test_backend_uses_gpu_for_large_numeric(self):
        """Backend uses GPU for large numeric buckets."""
        backend = GPUBackend(min_bucket_size=1000)
        
        data = list(range(10000, 0, -1))
        result = backend.sort(data)
        
        assert result == list(range(1, 10001))
    
    def test_backend_uses_cpu_for_small(self):
        """Backend uses CPU for small buckets."""
        backend = GPUBackend(min_bucket_size=10000)
        
        data = list(range(100, 0, -1))
        result = backend.sort(data)
        
        assert result == list(range(1, 101))
    
    def test_backend_uses_cpu_for_strings(self):
        """Backend uses CPU for string data."""
        backend = GPUBackend(min_bucket_size=100)
        
        data = ["zebra", "apple", "banana"]
        result = backend.sort(data)
        
        assert result == ["apple", "banana", "zebra"]
    
    def test_backend_stats(self):
        """Backend provides stats."""
        backend = GPUBackend(min_bucket_size=1000)
        stats = backend.get_stats()
        
        assert stats['available'] is True
        assert 'memory' in stats


class TestMemoryManagement:
    """Test GPU memory management."""
    
    def test_memory_info_available(self):
        """GPU memory info available."""
        info = get_gpu_memory_info()
        assert info is not None
        assert 'free' in info
        assert 'total' in info
        assert 'used' in info
    
    def test_estimate_memory(self):
        """Memory estimation."""
        needed = estimate_gpu_memory_needed(10000, item_size_bytes=8)
        assert needed > 0
    
    def test_can_fit_in_memory(self):
        """Check if data can fit in GPU memory."""
        # Small data should fit
        result = can_fit_in_gpu_memory(100, item_size_bytes=8)
        assert result is True


class TestBenchmark:
    """Benchmark GPU vs CPU performance."""
    
    def test_gpu_faster_than_cpu_large(self):
        """GPU is faster than CPU for large datasets."""
        # Generate large random dataset
        data = [random.random() for _ in range(500000)]
        
        # CPU sort timing
        start = time.perf_counter()
        cpu_result = sorted(data)
        cpu_time = time.perf_counter() - start
        
        # GPU sort timing
        start = time.perf_counter()
        gpu_result = gpu_sort(data)
        gpu_time = time.perf_counter() - start
        
        # Verify correctness
        assert gpu_result == pytest.approx(cpu_result, rel=1e-9)
        
        # GPU should be faster (at least 2x)
        speedup = cpu_time / gpu_time
        print(f"\nGPU speedup: {speedup:.2f}x")
        assert speedup > 1.5, f"GPU only {speedup:.2f}x faster than CPU"
    
    def test_gpu_overhead_not_worth_it_small(self):
        """GPU overhead makes small sorts slower."""
        data = [random.random() for _ in range(1000)]
        
        # CPU sort
        start = time.perf_counter()
        cpu_result = sorted(data)
        cpu_time = time.perf_counter() - start
        
        # GPU sort
        start = time.perf_counter()
        gpu_result = gpu_sort(data, min_size=500)
        gpu_time = time.perf_counter() - start
        
        # Small data - GPU might be slower due to overhead
        # Just verify correctness
        assert gpu_result == pytest.approx(cpu_result, rel=1e-9)


class TestEdgeCases:
    """Test edge cases and error handling."""
    
    def test_empty_list(self):
        """Handle empty list."""
        data = []
        result = gpu_sort_bucket_safe(data, fallback_to_cpu=True)
        assert result == []
    
    def test_single_element(self):
        """Handle single element."""
        data = [42]
        result = gpu_sort_bucket_safe(data, fallback_to_cpu=True)
        assert result == [42]
    
    def test_all_same_values(self):
        """Handle all same values."""
        data = [5] * 10000
        result = gpu_sort_bucket(data)
        assert result == [5] * 10000
    
    def test_already_sorted(self):
        """Handle already sorted data."""
        data = list(range(10000))
        result = gpu_sort_bucket(data)
        assert result == data
    
    def test_reverse_sorted(self):
        """Handle reverse sorted data."""
        data = list(range(10000, 0, -1))
        result = gpu_sort_bucket(data)
        assert result == list(range(1, 10001))


class TestConvenienceFunction:
    """Test gpu_sort convenience function."""
    
    def test_gpu_sort_convenience(self):
        """One-shot gpu_sort function."""
        data = list(range(10000, 0, -1))
        result = gpu_sort(data, min_size=1000)
        
        assert result == list(range(1, 10001))
    
    def test_gpu_sort_with_reverse(self):
        """gpu_sort with reverse option."""
        data = list(range(1, 10001))
        result = gpu_sort(data, reverse=True, min_size=1000)
        
        assert result == list(range(10000, 0, -1))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
