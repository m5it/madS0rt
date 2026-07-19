# madS0rt API Documentation

## Table of Contents

- [Core Sorter](#core-sorter)
- [Adaptive Sorter](#adaptive-sorter)
- [GPU Acceleration](#gpu-acceleration)
- [Extractors](#extractors)
- [Bucket Management](#bucket-management)
- [Hash Utilities](#hash-utilities)

---

## Core Sorter

### `MadSorter`

```python
MadSorter(
    prefix_length: int = 3,
    hash_provider: Union[str, HashProvider] = None,
    key_func: Optional[Callable[[Any], Any]] = None,
    strategy: Optional[SortStrategy] = None,
    max_bucket_size: Optional[int] = None,
    copy_mode: bool = False,
    use_gpu: bool = False,
    gpu_threshold: int = 10000,
)
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `prefix_length` | int | 3 | Number of characters to use for prefix bucketing |
| `hash_provider` | str or HashProvider | "crc32" | Hash function: 'crc32', 'xxhash32', 'xxhash64' |
| `key_func` | Callable | None | Function to extract sort key from items |
| `strategy` | SortStrategy | None | Algorithm selection strategy |
| `max_bucket_size` | int | None | Auto-split buckets larger than this |
| `copy_mode` | bool | False | If True, return new list; if False, sort in-place |
| `use_gpu` | bool | False | Enable GPU acceleration for large numeric buckets |
| `gpu_threshold` | int | 10000 | Minimum bucket size to use GPU (numeric data only) |

**GPU Acceleration:**

When `use_gpu=True`, MadSorter will automatically use GPU acceleration for large numeric buckets (> `gpu_threshold` items). This requires CuPy and a CUDA-capable GPU.

```python
# Enable GPU acceleration
sorter = MadSorter(
    use_gpu=True,
    gpu_threshold=10000  # Only use GPU for buckets > 10K items
)
```

**GPU Requirements:**
- CuPy installed (`pip install cupy-cuda11x` or `cupy-cuda12x`)
- CUDA-capable GPU
- Numeric data types only (int, float)

#### Methods

##### `sort(items: List[Any], reverse: bool = False) -> List[Any]`

Sort items using the madS0rt algorithm.

**Parameters:**
- `items`: List of items to sort
- `reverse`: If True, sort in descending order

**Returns:** Sorted list (or modified original if `copy_mode=False`)

**Example:**
```python
sorter = MadSorter(prefix_length=2)
result = sorter.sort(['banana', 'apple', 'cherry'])
# ['apple', 'banana', 'cherry']
```

##### `sorted(items: List[Any], reverse: bool = False) -> List[Any]`

Always returns a new sorted list (ignores `copy_mode`).

##### `get_stats() -> Dict[str, Any]`

Get timing and performance statistics.

**Returns:**
```python
{
    'total_items': int,
    'num_buckets': int,
    'sort_time_ms': float,
    'merge_time_ms': float,
    'total_time_ms': float,
    'bucket_stats': Dict,
    'gpu_enabled': bool,
    'gpu_available': bool,
    'gpu_buckets_sorted': int,
    'cpu_buckets_sorted': int,
}
```

##### `get_buckets() -> Dict[int, Bucket]`

Get the internal bucket distribution (for debugging/analysis).

##### `reset()`

Clear internal state and statistics.

---

### `SortStrategy`

Configuration for adaptive algorithm selection.

#### Constructor

```python
SortStrategy(
    insertion_threshold: int = 10,
    timsort_threshold: int = 1000,
    use_builtin_only: bool = False
)
```

**Algorithm Selection:**

| Bucket Size | Algorithm | Reason |
|-------------|-----------|--------|
| ≤ 10 | Insertion sort | Low overhead for tiny lists |
| 11 - 1000 | Timsort | Python's optimized hybrid sort |
| > 1000 | Hybrid | With bucket splitting |

---

## Adaptive Sorter

### `AdaptiveMadSorter`

Self-optimizing sorter that analyzes data distribution and adjusts parameters automatically.

#### Constructor

```python
AdaptiveMadSorter(
    initial_prefix_length: int = 3,
    hash_provider: Union[str, HashProvider] = None,
    key_func: Optional[Callable[[Any], Any]] = None,
    auto_adjust: bool = True,
    enable_load_balance: bool = True,
    max_bucket_size: Optional[int] = None
)
```

**Additional Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `auto_adjust` | bool | True | Analyze data and adjust prefix length |
| `enable_load_balance` | bool | True | Rebalance uneven bucket distributions |

#### Methods

##### `get_adaptive_report() -> str`

Get a human-readable report of optimization decisions.

**Example:**
```python
sorter = AdaptiveMadSorter(auto_adjust=True)
sorter.sort(large_dataset)
print(sorter.get_adaptive_report())
```

---

## GPU Acceleration

### `GPUBackend`

High-level interface for GPU-accelerated sorting.

#### Constructor

```python
GPUBackend(
    min_bucket_size: int = 10000,
    device_id: int = 0
)
```

#### Methods

##### `is_available() -> bool`

Check if GPU backend is available.

##### `should_use_gpu(data: List[Any]) -> bool`

Determine if GPU should be used for this data.

##### `sort(data: List[Any], reverse: bool = False, key_func: Optional[Callable] = None) -> List[Any]`

Sort data using GPU or CPU fallback.

### `gpu_sort(data, reverse=False, min_size=10000)`

One-shot GPU sort with automatic fallback.

```python
from madsort import gpu_sort

result = gpu_sort(large_numeric_list, min_size=10000)
```

### `gpu_available() -> bool`

Check if GPU acceleration is available.

```python
from madsort import gpu_available

if gpu_available():
    print("GPU ready!")
else:
    print("Install CuPy for GPU support")
```

---

## Extractors

All extractors are callable and can be used as `key_func` parameters.

### `FirstNCharsExtractor`

Extract first N characters from string keys.

```python
FirstNCharsExtractor(
    n: int = 3,
    lowercase: bool = True,
    normalize: bool = True,
    pad: Optional[str] = None,
    from_end: bool = False
)
```

### `NumericExtractor`

Extract numeric values from strings.

```python
NumericExtractor(
    pattern: Optional[str] = None,
    group: int = 1,
    as_type: type = int,
    default: Any = 0,
    allow_negative: bool = True,
    allow_decimal: bool = True
)
```

### `PathExtractor`

Extract values using deep path access like 'user.profile.name'.

```python
PathExtractor(
    path: Union[str, List[str]],
    separator: str = ".",
    default: Any = None
)
```

**Example:**
```python
from madsort import PathExtractor

# Dot notation
extractor = PathExtractor("user.profile.name")
data = {"user": {"profile": {"name": "Alice"}}}
extractor.extract(data)  # "Alice"

# Object attributes
extractor = PathExtractor("address.city")
person = Person(Address("Boston"))
extractor.extract(person)  # "Boston"
```

### `RegexExtractor`

Extract values using regular expressions.

```python
RegexExtractor(
    pattern: Union[str, Pattern],
    group: Union[int, str] = 0,
    fallback: Any = "",
    transform: Optional[Callable] = None,
    multiple: bool = False,
    separator: str = "|"
)
```

### `MultiFieldExtractor`

Combine multiple fields from dictionaries or objects.

```python
MultiFieldExtractor(
    fields: List[Union[str, Tuple[str, BaseExtractor]]],
    separator: str = "|",
    missing: str = "_",
    extractor: Optional[BaseExtractor] = None
)
```

### `CompositeKeyExtractor`

Create multi-level sort keys using tuples.

```python
CompositeKeyExtractor(
    extractors: List[BaseExtractor],
    priorities: Optional[List[int]] = None
)
```

**Example:**
```python
# Sort by category (str), then by priority (int), then by name
composite = CompositeKeyExtractor([
    lambda x: x['category'],
    NumericExtractor(default=999),
    lambda x: x['name']
])

items.sort(key=composite)
```

---

## Bucket Management

### `Bucket`

Single bucket containing items with similar prefixes.

#### Properties

| Property | Type | Description |
|----------|------|-------------|
| `bucket_id` | int | Unique identifier |
| `hash_key` | int | Hash value defining this bucket |
| `items` | List[Any] | Items in bucket |

#### Methods

##### `sort(reverse: bool = False, key: Optional[Callable] = None)`

Sort items in this bucket.

##### `merge(other: Bucket, preserve_order: bool = False) -> Bucket`

Merge another bucket into this one.

##### `split(predicate: Callable[[Any], bool]) -> Tuple[Bucket, Bucket]`

Split bucket based on predicate.

### `BucketManager`

Manages multiple buckets with automatic distribution.

#### Constructor

```python
BucketManager(
    hash_provider: Optional[HashProvider] = None,
    prefix_length: int = 3,
    key_func: Optional[Callable[[Any], str]] = None,
    max_bucket_size: Optional[int] = None
)
```

---

## Hash Utilities

### `crc32_hash(data: Union[str, bytes]) -> int`

Compute CRC32 hash (compatible with ptext.py).

### `xxhash_hash(data: Union[str, bytes], bits: int = 32, seed: int = 0) -> int`

Compute xxHash (faster alternative for large datasets).

### `get_hash_provider(name: str, **kwargs) -> HashProvider`

Factory function for hash providers.

---

## Convenience Functions

### `madsort(items, key=None, reverse=False, prefix_length=3, copy=True, use_gpu=False, gpu_threshold=10000)`

One-shot sorting function with optional GPU.

```python
from madsort import madsort

# With GPU acceleration
madsort(large_numeric_list, use_gpu=True, gpu_threshold=10000)
```

### `madsorted(items, key=None, reverse=False, prefix_length=3, use_gpu=False, gpu_threshold=10000)`

Always returns new sorted list.

```python
from madsort import madsorted

result = madsorted(large_list, use_gpu=True)
```

---

## Factory Functions

### `make_extractor(type_name: str, **kwargs) -> BaseExtractor`

Create extractors by type name.

```python
from madsort import make_extractor

# Available types: 'prefix', 'suffix', 'regex', 'numeric', 
#                  'path', 'multi_field', 'composite'

extractor = make_extractor('path', path='user.name')
```

### Preset Extractors

```python
from madsort import (
    make_filename_extractor,
    make_version_extractor,
    make_date_extractor
)

# Sort by name and extension
filename_ext = make_filename_extractor()

# Sort version strings (1.2.3 -> (1, 2, 3))
version_ext = make_version_extractor()

# Sort dates (YYYY-MM-DD)
date_ext = make_date_extractor()
```
