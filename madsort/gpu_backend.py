"""
GPU Backend for madS0rt.
Provides CUDA-accelerated sorting for large numeric buckets using CuPy.
"""

import logging
import warnings
from typing import List, Union, Optional, Any, Callable
import numpy as np

# Configure logging
logger = logging.getLogger(__name__)

# GPU availability flag
_GPU_AVAILABLE = None
_CUPY_MODULE = None


def gpu_available() -> bool:
    """
    Check if GPU acceleration is available.
    
    Returns:
        True if CuPy is installed and CUDA is available, False otherwise.
    """
    global _GPU_AVAILABLE, _CUPY_MODULE
    
    if _GPU_AVAILABLE is not None:
        return _GPU_AVAILABLE
    
    try:
        import cupy as cp
        # Test if CUDA is actually available
        cp.cuda.runtime.getDeviceCount()
        _CUPY_MODULE = cp
        _GPU_AVAILABLE = True
        logger.info("GPU acceleration available (CuPy + CUDA)")
        return True
    except ImportError:
        _GPU_AVAILABLE = False
        logger.debug("GPU not available: CuPy not installed")
        return False
    except Exception as e:
        _GPU_AVAILABLE = False
        logger.warning(f"GPU not available: {e}")
        return False


def get_cupy():
    """
    Get CuPy module if available.
    
    Returns:
        CuPy module or None if not available.
    """
    global _CUPY_MODULE
    if _CUPY_MODULE is None and gpu_available():
        import cupy as cp
        _CUPY_MODULE = cp
    return _CUPY_MODULE


def is_gpu_sortable(data: List[Any], min_size: int = 10000) -> bool:
    """
    Check if data is suitable for GPU sorting.
    
    Args:
        data: List of items to check
        min_size: Minimum size threshold for GPU
    
    Returns:
        True if data can be sorted on GPU, False otherwise.
    """
    # Check GPU availability
    if not gpu_available():
        return False
    
    # Check size
    if len(data) < min_size:
        return False
    
    # Check data type - must be homogeneous numeric
    if not data:
        return False
    
    # Check first item type
    first_item = data[0]
    
    # Only numeric types supported
    if isinstance(first_item, (int, float, np.number)):
        # Check all items are same type
        first_type = type(first_item)
        return all(isinstance(x, first_type) for x in data[:100])  # Sample check
    elif isinstance(first_item, np.ndarray):
        # NumPy arrays - check dtype
        return np.issubdtype(first_item.dtype, np.number)
    
    return False


def transfer_to_gpu(data: List[Union[int, float]], dtype: Optional[type] = None) -> Any:
    """
    Transfer data from CPU to GPU memory.
    
    Args:
        data: List of numeric values
        dtype: Optional NumPy dtype to convert to
    
    Returns:
        CuPy array on GPU.
    
    Raises:
        RuntimeError: If GPU not available.
    """
    cp = get_cupy()
    if cp is None:
        raise RuntimeError("GPU not available")
    
    # Convert to NumPy array first
    np_array = np.array(data, dtype=dtype)
    
    # Transfer to GPU
    try:
        gpu_array = cp.array(np_array)
        logger.debug(f"Transferred {len(data)} items to GPU")
        return gpu_array
    except cp.cuda.runtime.CUDARuntimeError as e:
        raise RuntimeError(f"Failed to transfer to GPU: {e}")


def transfer_to_cpu(gpu_array: Any) -> List[Union[int, float]]:
    """
    Transfer data from GPU to CPU memory.
    
    Args:
        gpu_array: CuPy array on GPU.
    
    Returns:
        List of values on CPU.
    """
    try:
        # Transfer from GPU to CPU
        np_array = gpu_array.get()
        # Convert to Python list
        result = np_array.tolist()
        logger.debug(f"Transferred {len(result)} items from GPU to CPU")
        return result
    except Exception as e:
        raise RuntimeError(f"Failed to transfer from GPU: {e}")


def gpu_sort_bucket(
    data: List[Union[int, float]],
    reverse: bool = False,
    key_func: Optional[Callable] = None
) -> List[Union[int, float]]:
    """
    Sort a bucket using GPU acceleration.
    
    Args:
        data: List of numeric values to sort
        reverse: If True, sort in descending order
        key_func: Optional key function (applied before sort)
    
    Returns:
        Sorted list of values.
    
    Raises:
        RuntimeError: If GPU sorting fails.
        ValueError: If data not suitable for GPU.
    """
    cp = get_cupy()
    if cp is None:
        raise RuntimeError("GPU not available")
    
    # Check if data is suitable
    if not is_gpu_sortable(data):
        raise ValueError("Data not suitable for GPU sorting")
    
    try:
        # Apply key function if provided
        if key_func:
            # Key function must return numeric values
            keyed_data = [key_func(x) for x in data]
            if not all(isinstance(x, (int, float, np.number)) for x in keyed_data[:100]):
                raise ValueError("Key function must return numeric values for GPU")
            gpu_array = transfer_to_gpu(keyed_data)
        else:
            gpu_array = transfer_to_gpu(data)
        
        # Sort on GPU using CuPy
        # CuPy uses optimized radix sort for integers, merge sort for floats
        sorted_gpu = cp.sort(gpu_array)
        
        if reverse:
            sorted_gpu = sorted_gpu[::-1]
        
        # Transfer back to CPU
        result = transfer_to_cpu(sorted_gpu)
        
        logger.debug(f"GPU sorted {len(result)} items")
        return result
        
    except Exception as e:
        logger.error(f"GPU sort failed: {e}")
        raise


def gpu_sort_bucket_safe(
    data: List[Any],
    reverse: bool = False,
    key_func: Optional[Callable] = None,
    fallback_to_cpu: bool = True
) -> List[Any]:
    """
    Sort a bucket using GPU with automatic CPU fallback.
    
    Args:
        data: List of values to sort
        reverse: If True, sort in descending order
        key_func: Optional key function
        fallback_to_cpu: If True, fall back to CPU on GPU failure
    
    Returns:
        Sorted list of values.
    """
    # Check if we should even try GPU
    if not is_gpu_sortable(data):
        logger.debug("Data not GPU-sortable, using CPU")
        return _cpu_sort(data, reverse, key_func)
    
    try:
        return gpu_sort_bucket(data, reverse, key_func)
    except Exception as e:
        if fallback_to_cpu:
            logger.warning(f"GPU sort failed ({e}), falling back to CPU")
            return _cpu_sort(data, reverse, key_func)
        else:
            raise


def _cpu_sort(
    data: List[Any],
    reverse: bool = False,
    key_func: Optional[Callable] = None
) -> List[Any]:
    """
    Fallback CPU sort.
    
    Args:
        data: List to sort
        reverse: Reverse order
        key_func: Key function
    
    Returns:
        Sorted list.
    """
    result = data.copy()
    result.sort(key=key_func, reverse=reverse)
    return result


def get_gpu_memory_info() -> Optional[dict]:
    """
    Get GPU memory information.
    
    Returns:
        Dictionary with memory info or None if GPU not available.
    """
    cp = get_cupy()
    if cp is None:
        return None
    
    try:
        mem_info = cp.cuda.Device().mem_info
        return {
            'free': mem_info[0],
            'total': mem_info[1],
            'used': mem_info[1] - mem_info[0],
        }
    except Exception as e:
        logger.warning(f"Failed to get GPU memory info: {e}")
        return None


def estimate_gpu_memory_needed(n_items: int, item_size_bytes: int = 8) -> int:
    """
    Estimate GPU memory needed for sorting.
    
    Args:
        n_items: Number of items to sort
        item_size_bytes: Size of each item in bytes
    
    Returns:
        Estimated memory in bytes.
    """
    # Need: original array + sorted array + temporary workspace
    # Rough estimate: 3x the data size
    return n_items * item_size_bytes * 3


def can_fit_in_gpu_memory(n_items: int, item_size_bytes: int = 8) -> bool:
    """
    Check if data can fit in GPU memory.
    
    Args:
        n_items: Number of items
        item_size_bytes: Size of each item
    
    Returns:
        True if data can fit in GPU memory.
    """
    mem_info = get_gpu_memory_info()
    if mem_info is None:
        return False
    
    needed = estimate_gpu_memory_needed(n_items, item_size_bytes)
    return needed < mem_info['free'] * 0.8  # Leave 20% headroom


class GPUBackend:
    """
    High-level GPU backend interface for madS0rt.
    
    Provides convenient methods for GPU-accelerated sorting with
    automatic fallback and memory management.
    """
    
    def __init__(
        self,
        min_bucket_size: int = 10000,
        device_id: int = 0
    ):
        """
        Initialize GPU backend.
        
        Args:
            min_bucket_size: Minimum bucket size to use GPU
            device_id: CUDA device ID to use
        """
        self.min_bucket_size = min_bucket_size
        self.device_id = device_id
        self._cp = get_cupy()
        
        if self._cp and device_id > 0:
            try:
                self._cp.cuda.Device(device_id).use()
                logger.info(f"Using GPU device {device_id}")
            except Exception as e:
                logger.warning(f"Failed to set GPU device {device_id}: {e}")
    
    def is_available(self) -> bool:
        """Check if GPU backend is available."""
        return self._cp is not None
    
    def should_use_gpu(self, data: List[Any]) -> bool:
        """
        Determine if GPU should be used for this data.
        
        Args:
            data: Data to sort
        
        Returns:
            True if GPU should be used.
        """
        if not self.is_available():
            return False
        
        if len(data) < self.min_bucket_size:
            return False
        
        return is_gpu_sortable(data, self.min_bucket_size)
    
    def sort(
        self,
        data: List[Any],
        reverse: bool = False,
        key_func: Optional[Callable] = None
    ) -> List[Any]:
        """
        Sort data using GPU or CPU fallback.
        
        Args:
            data: Data to sort
            reverse: Reverse order
            key_func: Key function
        
        Returns:
            Sorted data.
        """
        if self.should_use_gpu(data):
            return gpu_sort_bucket_safe(data, reverse, key_func, fallback_to_cpu=True)
        else:
            return _cpu_sort(data, reverse, key_func)
    
    def get_stats(self) -> dict:
        """
        Get GPU backend statistics.
        
        Returns:
            Dictionary with stats.
        """
        return {
            'available': self.is_available(),
            'min_bucket_size': self.min_bucket_size,
            'device_id': self.device_id,
            'memory': get_gpu_memory_info(),
        }


# Convenience function for one-shot GPU sorting
def gpu_sort(
    data: List[Union[int, float]],
    reverse: bool = False,
    min_size: int = 10000
) -> List[Union[int, float]]:
    """
    One-shot GPU sort with automatic fallback.
    
    Args:
        data: Numeric data to sort
        reverse: Reverse order
        min_size: Minimum size to use GPU
    
    Returns:
        Sorted data.
    """
    backend = GPUBackend(min_bucket_size=min_size)
    return backend.sort(data, reverse)
