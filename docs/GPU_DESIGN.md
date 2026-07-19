# GPU Acceleration Design for madS0rt

## Executive Summary

This document outlines the architecture for adding optional GPU acceleration to madS0rt, enabling high-performance sorting of large numeric datasets using CUDA-enabled GPUs.

## When to Use GPU

### Criteria for GPU Acceleration

| Factor | Threshold | Reason |
|--------|-----------|--------|
| **Bucket Size** | > 10,000 items | Memory transfer overhead amortization |
| **Data Type** | Numeric only (int, float) | GPU excels at parallel numeric operations |
| **Data Homogeneity** | Single type per bucket | Avoids costly type checking on GPU |
| **GPU Memory** | < 80% available | Prevents out-of-memory errors |
| **Sort Type** | Comparison-based | Radix sort on GPU is extremely fast |

### When NOT to Use GPU

- Small buckets (< 10K items) - CPU overhead lower than transfer cost
- String data - Variable length, complex comparison
- Mixed types - Type conversion overhead
- Nested objects - Memory layout too complex
- Limited GPU memory - Fallback to CPU

## Technology Selection

### Chosen: CuPy

**Why CuPy over PyCUDA or Numba:**
- **NumPy-compatible API** - Minimal code changes
- **Built-in sorting** - `cupy.sort()` uses optimized CUDA radix sort
- **Memory management** - Automatic pool allocation, less boilerplate
- **Mature ecosystem** - Well-maintained, extensive documentation
- **Easy installation** - `pip install cupy-cuda11x` (matching CUDA version)

### Comparison

| Library | Pros | Cons | Verdict |
|---------|------|------|---------|
| **CuPy** | NumPy API, built-in sort, easy install | Requires matching CUDA version | ✅ **Chosen** |
| **PyCUDA** | Full control, flexible | Verbose, manual memory management | ❌ Too complex |
| **Numba CUDA** | JIT compilation, Python syntax | Compilation overhead, less mature | ❌ Overkill |

## Hybrid CPU/GPU Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    MadSorter with GPU                        │
├─────────────────────────────────────────────────────────────┤
│  Phase 1: DISTRIBUTE (CPU)                                   │
│  ├─ Extract prefix → hash → bucket (always CPU)             │
│  └─ Result: Buckets with mixed sizes                        │
│                                                              │
│  Phase 2: SORT (Hybrid CPU/GPU)                              │
│  ├─ Small bucket (≤10K) → CPU insertion sort                │
│  ├─ Medium bucket (10K-100K) → CPU Timsort                │
│  └─ Large numeric bucket (>100K) → GPU radix sort           │
│      ├─ Transfer to GPU memory                               │
│      ├─ cuPy.sort() (parallel radix sort)                   │
│      └─ Transfer back to CPU memory                         │
│                                                              │
│  Phase 3: MERGE (CPU)                                       │
│  └─ K-way merge of all sorted buckets (CPU)                │
└─────────────────────────────────────────────────────────────┘
```

## Memory Transfer Optimization

### Strategy 1: Pinned Memory (Page-Locked)

```python
# Use pinned memory for faster CPU→GPU transfers
import cupy as cp

# Allocate pinned memory
pinned = cp.cuda.alloc_pinned_memory(size)
# Async transfer
cp.cuda.stream.get_current_stream().memcpy_async(gpu_array, pinned, size)
```

### Strategy 2: Memory Pool

```python
# CuPy's memory pool reduces allocation overhead
pool = cp.cuda.MemoryPool(cp.cuda.malloc_managed)
cp.cuda.set_allocator(pool.malloc)
```

### Strategy 3: Unified Memory (Optional)

```python
# Managed memory accessible by both CPU and GPU
# Simpler but potentially slower
managed_array = cp.cuda.malloc_managed(size)
```

### Strategy 4: Batch Transfer

```python
# Sort multiple buckets on GPU before transferring back
# Reduces transfer overhead
gpu_buckets = []
for bucket in large_numeric_buckets:
    gpu_buckets.append(transfer_to_gpu(bucket))
    
# Sort all on GPU
for gpu_bucket in gpu_buckets:
    sort_on_gpu(gpu_bucket)
    
# Transfer all back
for gpu_bucket in gpu_buckets:
    transfer_to_cpu(gpu_bucket)
```

## Implementation Plan

### Module Structure

```
madsort/
├── gpu_backend.py          # GPU abstraction layer
│   ├── gpu_available()     # Check CUDA availability
│   ├── is_gpu_sortable()   # Check if data suitable for GPU
│   ├── gpu_sort_bucket()   # Sort bucket on GPU
│   └── transfer_*()        # Memory transfer functions
├── gpu_sorter.py           # GPU-enabled MadSorter variant
└── sorter.py               # Modified to support GPU option
```

### API Design

```python
class MadSorter:
    def __init__(
        self,
        prefix_length: int = 3,
        ...,
        use_gpu: bool = False,           # NEW: Enable GPU
        gpu_threshold: int = 10000,      # NEW: Min bucket size for GPU
        gpu_device: int = 0,             # NEW: GPU device ID
    ):
        ...
```

### Error Handling Strategy

```python
def gpu_sort_bucket(bucket):
    try:
        # Attempt GPU sort
        return _gpu_sort_internal(bucket)
    except (cp.cuda.runtime.CUDARuntimeError, 
            cp.cuda.memory.OutOfMemoryError) as e:
        # Log warning and fallback to CPU
        logger.warning(f"GPU sort failed: {e}, falling back to CPU")
        return cpu_sort_bucket(bucket)
```

## Performance Expectations

### Expected Speedups

| Data Size | GPU Speedup vs CPU | Notes |
|-----------|-------------------|-------|
| 100K items | 5-10x | Good GPU utilization |
| 1M items | 10-50x | Excellent GPU utilization |
| 10M items | 20-100x | Memory bandwidth limited |
| 100M items | 50-200x | PCIe bandwidth bottleneck |

### Bottlenecks

1. **PCIe Transfer** - CPU↔GPU memory transfer is main bottleneck
   - Mitigation: Batch transfers, keep data on GPU longer
   
2. **Kernel Launch** - Small buckets waste GPU parallelism
   - Mitigation: Only use GPU for buckets > 10K items
   
3. **Memory Allocation** - Dynamic allocation slows GPU
   - Mitigation: Use memory pools

## Testing Strategy

### Unit Tests
- GPU availability detection
- Memory transfer functions
- Sort correctness vs CPU
- Error handling and fallback

### Integration Tests
- Hybrid sorting (mixed CPU/GPU buckets)
- Large dataset sorting
- Memory limit handling

### Benchmarks
- Compare GPU vs CPU for various sizes
- Measure transfer overhead
- Profile memory usage

## Installation Requirements

### Optional Dependency

GPU support is optional. Install with:

```bash
# CPU only (default)
pip install mads0rt

# With GPU support
pip install mads0rt[gpu]

# Or manually install CuPy matching your CUDA version
pip install cupy-cuda11x  # For CUDA 11.x
pip install cupy-cuda12x  # For CUDA 12.x
```

### Runtime Detection

```python
try:
    import cupy as cp
    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False
```

## Limitations and Future Work

### Current Limitations

1. **Numeric only** - Strings, objects not supported on GPU
2. **CUDA only** - No OpenCL, ROCm support yet
3. **Single GPU** - No multi-GPU support yet
4. **Comparison sort only** - No custom comparators on GPU

### Future Enhancements

1. **Multi-GPU** - Distribute buckets across multiple GPUs
2. **Custom kernels** - Write custom CUDA kernels for specific use cases
3. **Async sorting** - Overlap CPU and GPU sorting
4. **Unified memory** - Simplify memory management
5. **ROCm support** - AMD GPU compatibility

## Conclusion

GPU acceleration will provide significant speedup (10-100x) for large numeric sorting tasks while maintaining madS0rt's hybrid architecture. The CuPy-based approach minimizes code changes and provides robust fallback to CPU when GPU is unavailable or unsuitable.
