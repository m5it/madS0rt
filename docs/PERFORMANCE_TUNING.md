# Performance Tuning Guide for madS0rt

## Quick Reference

| Scenario | Recommended Settings | Expected Speedup |
|----------|---------------------|------------------|
| Small data (<1K items) | `prefix_length=2`, adaptive=False | 0.8x - 1.0x |
| Medium data (1K-100K) | `prefix_length=3`, adaptive=True | 1.2x - 2.0x |
| Large data (>100K) | `prefix_length=3-4`, adaptive=True | 2.0x - 5.0x |
| Clustered data | `prefix_length=4-5`, load_balance=True | 3.0x - 10x |
| Few unique prefixes | `prefix_length=1-2` | 1.5x - 3.0x |

## Understanding Your Data

### 1. Distribution Analysis

Before tuning, analyze your data distribution:

```python
from madsort import DistributionAnalyzer

analyzer = DistributionAnalyzer()
stats = analyzer.analyze(your_data, key_func=lambda x: str(x))

print(f"Distribution type: {stats['distribution_type']}")
print(f"Optimal prefix length: {stats['optimal_prefix_length']}")
print(f"Entropy: {stats['optimal_stats']['entropy']:.2f}")

for rec in analyzer.get_recommendations():
    print(f"  - {rec}")
```

### 2. Distribution Types

| Type | Characteristics | Strategy |
|------|----------------|----------|
| **Uniform** | Even distribution, high entropy | Standard settings work well |
| **Dense** | Many items, few unique prefixes | Increase prefix length |
| **Sparse** | Few items, many unique prefixes | Decrease prefix length |
| **Skewed** | Heavy concentration in few buckets | Enable load balancing |

## Prefix Length Selection

### Guidelines by Data Type

```python
# English words - 2-3 chars good balance
sorter = MadSorter(prefix_length=3)

# Random strings - need more chars to differentiate
sorter = MadSorter(prefix_length=4)

# Similar prefixes (URLs, paths) - more chars needed
sorter = MadSorter(prefix_length=5)

# Numeric IDs - prefix doesn't help, use full key
sorter = MadSorter(prefix_length=10)  # Or use adaptive
```

### Testing Different Prefix Lengths

```python
from madsort import MadSorter
import time

def benchmark_prefix_lengths(data, lengths=[2, 3, 4, 5]):
    """Find optimal prefix length for your data."""
    results = []
    
    for length in lengths:
        sorter = MadSorter(prefix_length=length)
        
        start = time.perf_counter()
        sorter.sort(data.copy())
        elapsed = (time.perf_counter() - start) * 1000
        
        stats = sorter.get_stats()
        results.append({
            'prefix_length': length,
            'time_ms': elapsed,
            'buckets': stats['num_buckets']
        })
    
    return results

# Usage
data = [...]  # Your data
results = benchmark_prefix_lengths(data)
for r in results:
    print(f"Length {r['prefix_length']}: {r['time_ms']:.2f}ms, "
          f"{r['buckets']} buckets")
```

## Memory Optimization

### For Large Datasets

```python
# Stream processing for very large files
def sort_large_file(input_path, output_path, chunk_size=10000):
    """Sort file in chunks to minimize memory usage."""
    from madsort import MadSorter
    
    # Read and sort chunks
    chunks = []
    current_chunk = []
    
    with open(input_path, 'r') as f:
        for line in f:
            current_chunk.append(line.rstrip())
            
            if len(current_chunk) >= chunk_size:
                sorter = MadSorter()
                sorted_chunk = sorter.sort(current_chunk)
                chunks.append(sorted_chunk)
                current_chunk = []
        
        # Don't forget last chunk
        if current_chunk:
            sorter = MadSorter()
            chunks.append(sorter.sort(current_chunk))
    
    # K-way merge of sorted chunks
    import heapq
    with open(output_path, 'w') as f:
        for line in heapq.merge(*chunks):
            f.write(line + '\n')
```

### In-Place Sorting

```python
# When memory is tight, sort in-place
data = [...]  # Large list
madsort(data, copy=False)  # Modifies original, no duplication
```

## Adaptive Mode Deep Dive

### When to Use Adaptive Mode

✅ **Use when:**
- Data distribution unknown
- Mixed data types
- Very large datasets (1M+ items)
- Performance-critical applications

❌ **Avoid when:**
- Small, consistent datasets
- Performance predictable
- Overhead not worth it (< 10K items)

### Adaptive Configuration

```python
from madsort import AdaptiveMadSorter

sorter = AdaptiveMadSorter(
    initial_prefix_length=3,    # Starting point
    auto_adjust=True,             # Analyze and optimize
    enable_load_balance=True,     # Handle skewed data
    max_bucket_size=10000         # Split large buckets
)

result = sorter.sort(data)

# Review decisions
print(sorter.get_adaptive_report())
```

## Hash Provider Selection

### CRC32 vs xxHash

| Provider | Speed | Quality | Use Case |
|----------|-------|---------|----------|
| CRC32 | Fast | Good | Default, compatible with ptext.py |
| xxHash32 | Very Fast | Excellent | Large datasets |
| xxHash64 | Very Fast | Excellent | 64-bit systems |

```python
# xxHash for maximum speed
sorter = MadSorter(hash_provider='xxhash32')

# CRC32 for compatibility
sorter = MadSorter(hash_provider='crc32')
```

## Key Function Optimization

### Avoid Expensive Operations

```python
# BAD: Expensive operation in key function
def bad_key(item):
    return expensive_calculation(item)  # Called O(n log n) times

# GOOD: Pre-compute once
precomputed = {item: expensive_calculation(item) for item in data}
sorter = MadSorter(key_func=lambda x: precomputed[x])
```

### Use Appropriate Extractors

```python
from madsort import (
    NumericExtractor,
    RegexExtractor,
    CompositeKeyExtractor
)

# Numeric sorting
items = ["item_10", "item_2", "item_1"]
sorter = MadSorter(key_func=NumericExtractor())

# Regex extraction
sorter = MadSorter(key_func=RegexExtractor(r'(\d{4}-\d{2}-\d{2})'))

# Multi-level sorting
composite = CompositeKeyExtractor([
    lambda x: x.category,
    NumericExtractor(),
    lambda x: x.name
])
```

## Benchmarking Your Setup

### Quick Benchmark

```python
from madsort.benchmark import run_quick_benchmark

results = run_quick_benchmark()
```

### Custom Benchmark

```python
from madsort import BenchmarkSuite

suite = BenchmarkSuite(iterations=5, warmup=2)

# Your specific data
data = [...]

# Compare algorithms
py_result = suite.benchmark_python_sorted(data)
ms_result = suite.benchmark_mads0rt(data, prefix_length=3)

speedup = py_result.avg_time_ms / ms_result.avg_time_ms
print(f"Speedup: {speedup:.2f}x")
```

## Troubleshooting Performance

### Too Many Buckets

**Symptom:** High overhead, slow performance
**Cause:** Prefix length too long for data
**Fix:** Reduce `prefix_length` or use adaptive mode

```python
# Check bucket count
sorter = MadSorter(prefix_length=4)
sorter._bucket_manager.distribute(data)
print(f"Buckets: {len(sorter.get_buckets())}")  # Should be < 1000

# If too many, reduce prefix length
sorter = MadSorter(prefix_length=2)
```

### Too Few Buckets

**Symptom:** Buckets too large, no benefit
**Cause:** Prefix length too short
**Fix:** Increase `prefix_length` or enable load balancing

```python
# Check bucket sizes
stats = sorter.get_stats()
bucket_stats = stats.get('bucket_stats', {})
print(f"Largest bucket: {bucket_stats.get('largest_bucket', 'N/A')}")

# If too large, increase prefix or use load balancing
sorter = MadSorter(prefix_length=4, max_bucket_size=1000)
```

### Uneven Distribution

**Symptom:** Some buckets much larger than others
**Cause:** Skewed data
**Fix:** Enable load balancing

```python
from madsort import AdaptiveMadSorter

sorter = AdaptiveMadSorter(
    enable_load_balance=True,  # Rebalance uneven buckets
    auto_adjust=True
)
```

## Best Practices Summary

1. **Start with adaptive mode** for unknown data
2. **Analyze first** - use DistributionAnalyzer
3. **Tune prefix length** based on data characteristics
4. **Use appropriate hash** - xxHash for speed, CRC32 for compatibility
5. **Enable load balancing** for skewed data
6. **Benchmark** before and after changes
7. **Consider memory** - use in-place for large datasets
8. **Optimize key functions** - avoid expensive operations

## Example: Complete Tuning Workflow

```python
from madsort import (
    DistributionAnalyzer,

## GPU Acceleration Tuning

For large-scale numeric sorting, GPU acceleration can provide 10-100x speedup.

### When to Use GPU

✅ **Use GPU when:**
- Dataset > 100K numeric items
- Homogeneous numeric data (int, float)
- Sufficient GPU memory available
- CUDA-capable GPU available

❌ **Avoid GPU when:**
- Mixed data types (strings, objects)
- Small datasets (< 10K items) - transfer overhead
- Limited GPU memory
- Non-NVIDIA GPU

### GPU Configuration

```python
from madsort import MadSorter, gpu_available

# Check GPU availability
if gpu_available():
    print("GPU ready!")

# Configure for GPU
sorter = MadSorter(
    use_gpu=True,
    gpu_threshold=10000  # Only for buckets > 10K
)

# Sort large numeric dataset
data = [random.random() for _ in range(1000000)]
result = sorter.sort(data)
```

### GPU Threshold Tuning

| Threshold | Use Case |
|-----------|----------|
| 5,000 | Aggressive GPU usage (more transfers) |
| 10,000 | Balanced (default) |
| 50,000 | Conservative (only very large buckets) |
| 100,000 | Minimal GPU usage |

### GPU Memory Management

```python
from madsort.gpu_backend import (
    get_gpu_memory_info,
    estimate_gpu_memory_needed
)

# Check available memory
mem_info = get_gpu_memory_info()
print(f"Free GPU memory: {mem_info['free'] / 1e9:.2f} GB")

# Estimate before sorting
needed = estimate_gpu_memory_needed(1000000, item_size_bytes=8)
print(f"Estimated need: {needed / 1e9:.2f} GB")
```

### Hybrid CPU/GPU Strategy

```python
# For mixed data, GPU sorts numeric, CPU sorts strings
sorter = MadSorter(
    use_gpu=True,
    gpu_threshold=10000
)

# Numeric data → GPU
# String data → CPU (automatic fallback)
mixed_data = numbers + strings
result = sorter.sort(mixed_data)
```

### Expected Speedups

| Data Size | CPU Time | GPU Time | Speedup |
|-----------|----------|----------|---------|
| 100K | 0.05s | 0.01s | 5x |
| 1M | 0.5s | 0.05s | 10x |
| 10M | 5s | 0.2s | 25x |
| 100M | 60s | 1s | 60x |

*Results vary based on GPU model and PCIe bandwidth.*

## Additional Resources

- See `examples/performance_tuning.py` for working examples
- See `examples/gpu_acceleration.py` for GPU examples
- Run `python -m madsort.benchmark --full` for comprehensive benchmarks
- Check API documentation in `docs/API.md`
    
    # Step 1: Analyze
    analyzer = DistributionAnalyzer()
    analysis = analyzer.analyze(data, lambda x: str(x))
    print(f"Distribution: {analysis['distribution_type']}")
    
    # Step 2: Configure
    optimal_length = analysis['optimal_prefix_length']
    sorter = AdaptiveMadSorter(
        initial_prefix_length=optimal_length,
        auto_adjust=True,
        enable_load_balance=True
    )
    
    # Step 3: Benchmark
    suite = BenchmarkSuite(iterations=3)
    
    py_result = suite.benchmark_python_sorted(data)
    ms_result = suite.benchmark_mads0rt(
        data, 
        prefix_length=optimal_length
    )
    
    speedup = py_result.avg_time_ms / ms_result.avg_time_ms
    
    # Step 4: Report
    print(f"Optimal prefix length: {optimal_length}")
    print(f"Speedup over Python: {speedup:.2f}x")
    print(sorter.get_adaptive_report())
    
    return sorter

# Usage
data = [...]  # Your data
sorter = tune_for_dataset(data)
```

## Additional Resources

- See `examples/performance_tuning.py` for working examples
- Run `python -m madsort.benchmark --full` for comprehensive benchmarks
- Check API documentation in `docs/API.md`
