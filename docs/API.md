# madS0rt API Documentation

## Table of Contents

- [Core Sorter](#core-sorter)
- [Adaptive Sorter](#adaptive-sorter)
- [Extractors](#extractors)
- [Bucket Management](#bucket-management)
- [Hash Utilities](#hash-utilities)

---

## Core Sorter

### `MadSorter`

Main sorting engine implementing the hybrid bucket-radix sort algorithm.

#### Constructor

```python
MadSorter(
    prefix_length: int = 3,
    hash_provider: Union[str, HashProvider] = None,
    key_func: Optional[Callable[[Any], Any]] = None,
    strategy: Optional[SortStrategy] = None,
    max_bucket_size: Optional[int] = None,
    copy_mode: bool = False
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
    'bucket_stats': Dict
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
# Adaptive Sorting Report:
# =========================
# 
# Sort #1:
#   Items: 100000
#   Prefix length: 3
#   Distribution: uniform
#   Optimal length detected: 3
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

**Example:**
```python
extractor = FirstNCharsExtractor(n=4, lowercase=True)
extractor("HELLO")  # "hell"
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

**Example:**
```python
ext = NumericExtractor()
ext("item_123")      # 123
ext("price_12.99")   # 12 (or 12.99 with allow_decimal=True)
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

**Example:**
```python
# Extract date
date_ext = RegexExtractor(r'(\d{4}-\d{2}-\d{2})', group=1)
date_ext("Event: 2024-03-15")  # "2024-03-15"

# Extract with transform
num_ext = RegexExtractor(r'(\d+)', transform=int)
num_ext("abc123")  # 123 (as int, not str)
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

**Example:**
```python
extractor = MultiFieldExtractor(
    fields=['category', 'priority', 'name'],
    separator='-'
)

data = {'category': 'A', 'priority': '1', 'name': 'task'}
extractor(data)  # "A-1-task"
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

#### Methods

##### `distribute(items: List[Any]) -> Dict[int, Bucket]`

Distribute items into buckets based on prefix hash.

##### `add(item: Any) -> int`

Add single item to appropriate bucket.

##### `get_all_items(sorted_buckets: bool = False) -> List[Any]`

Retrieve all items from all buckets.

##### `get_stats() -> Dict[str, Any]`

Get bucket statistics.

---

## Hash Utilities

### `crc32_hash(data: Union[str, bytes]) -> int`

Compute CRC32 hash (compatible with ptext.py).

### `xxhash_hash(data: Union[str, bytes], bits: int = 32, seed: int = 0) -> int`

Compute xxHash (faster alternative for large datasets).

### `get_hash_provider(name: str, **kwargs) -> HashProvider`

Factory function for hash providers.

**Example:**
```python
from madsort import get_hash_provider

# CRC32 (default)
crc32 = get_hash_provider("crc32")

# xxHash 64-bit
xxh64 = get_hash_provider("xxhash64")
```

---

## Convenience Functions

### `madsort(items, key=None, reverse=False, prefix_length=3, copy=True)`

One-shot sorting function.

```python
from madsort import madsort

# Like list.sort() but with madS0rt algorithm
madsort(my_list, key=len, reverse=True, copy=False)
```

### `madsorted(items, key=None, reverse=False, prefix_length=3)`

Always returns new sorted list.

```python
from madsort import madsorted

result = madsorted(words, key=str.lower)
```

---

## Factory Functions

### `make_extractor(type_name: str, **kwargs) -> BaseExtractor`

Create extractors by type name.

```python
from madsort import make_extractor

# Available types: 'prefix', 'suffix', 'regex', 'numeric', 
#                  'multi_field', 'composite'

extractor = make_extractor('prefix', n=4)
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
