"""
Core bucket implementation for madS0rt.
Provides Bucket class and BucketManager for organizing data into hash-based buckets.
"""

from typing import List, Dict, Any, Optional, Callable, Iterator, Union, Tuple
from collections import defaultdict

from .hash_utils import HashProvider, CRC32Provider, get_hash_provider


class Bucket:
    """
    A single bucket containing items with similar key prefixes.
    Optimized for both small and large bucket sizes.
    """
    
    __slots__ = ['bucket_id', 'hash_key', 'items', '_sorted', '_key_func']
    
    def __init__(self, bucket_id: int, hash_key: int, key_func: Optional[Callable] = None):
        """
        Initialize a bucket.
        
        Args:
            bucket_id: Unique identifier for this bucket
            hash_key: The hash value that defines this bucket
            key_func: Optional key extraction function for sorting
        """
        self.bucket_id = bucket_id
        self.hash_key = hash_key
        self.items: List[Any] = []
        self._sorted = False
        self._key_func = key_func or (lambda x: x)
    
    def add(self, item: Any) -> None:
        """Add an item to the bucket."""
        self.items.append(item)
        self._sorted = False
    
    def add_batch(self, items: List[Any]) -> None:
        """Add multiple items efficiently."""
        self.items.extend(items)
        self._sorted = False
    
    def sort(self, reverse: bool = False, key: Optional[Callable] = None) -> None:
        """
        Sort items in this bucket.
        Uses insertion sort for small buckets, Timsort for larger ones.
        
        Args:
            reverse: Sort in descending order if True
            key: Optional key function (overrides default)
        """
        if self._sorted:
            return
        
        sort_key = key or self._key_func
        
        # For very small buckets (<= 10), insertion sort is efficient
        # Python's sort uses Timsort which handles this automatically
        self.items.sort(key=sort_key, reverse=reverse)
        self._sorted = True
    
    def merge(self, other: 'Bucket', preserve_order: bool = False) -> 'Bucket':
        """
        Merge another bucket into this one.
        
        Args:
            other: Bucket to merge from
            preserve_order: If True, maintains sorted order if both are sorted
        
        Returns:
            Self for chaining
        """
        if preserve_order and self._sorted and other._sorted:
            # Efficient merge of two sorted lists
            merged = []
            i = j = 0
            len_self = len(self.items)
            len_other = len(other.items)
            
            while i < len_self and j < len_other:
                if self.items[i] <= other.items[j]:
                    merged.append(self.items[i])
                    i += 1
                else:
                    merged.append(other.items[j])
                    j += 1
            
            # Append remaining items
            if i < len_self:
                merged.extend(self.items[i:])
            if j < len_other:
                merged.extend(other.items[j:])
            
            self.items = merged
        else:
            self.items.extend(other.items)
            self._sorted = False
        
        return self
    
    def split(self, predicate: Callable[[Any], bool]) -> Tuple['Bucket', 'Bucket']:
        """
        Split bucket into two based on predicate.
        
        Args:
            predicate: Function returning True for items to keep in first bucket
        
        Returns:
            Tuple of (matching_bucket, non_matching_bucket)
        """
        matching = Bucket(self.bucket_id * 2, self.hash_key, self._key_func)
        non_matching = Bucket(self.bucket_id * 2 + 1, self.hash_key, self._key_func)
        
        for item in self.items:
            if predicate(item):
                matching.add(item)
            else:
                non_matching.add(item)
        
        return matching, non_matching
    
    def __len__(self) -> int:
        return len(self.items)
    
    def __iter__(self) -> Iterator[Any]:
        return iter(self.items)
    
    def __bool__(self) -> bool:
        return len(self.items) > 0
    
    def __repr__(self) -> str:
        return f"Bucket(id={self.bucket_id}, hash={self.hash_key:08x}, items={len(self.items)})"


class BucketManager:
    """
    Manages multiple buckets with automatic distribution based on hash prefixes.
    Core of the madS0rt algorithm - groups similar items to reduce comparison scope.
    """
    
    def __init__(
        self,
        hash_provider: Optional[HashProvider] = None,
        prefix_length: int = 3,
        key_func: Optional[Callable[[Any], str]] = None,
        max_bucket_size: Optional[int] = None
    ):
        """
        Initialize bucket manager.
        
        Args:
            hash_provider: Hash function provider (default: CRC32)
            prefix_length: Number of characters to use for bucketing
            key_func: Function to extract string key from items
            max_bucket_size: Optional limit for bucket splitting
        """
        self.hash_provider = hash_provider or CRC32Provider()
        self.prefix_length = prefix_length
        self.key_func = key_func or self._default_key_func
        self.max_bucket_size = max_bucket_size
        
        self.buckets: Dict[int, Bucket] = {}
        self._bucket_counter = 0
        self._stats = {
            'total_items': 0,
            'num_buckets': 0,
            'largest_bucket': 0,
            'smallest_bucket': float('inf'),
            'empty_buckets': 0
        }
    
    def _default_key_func(self, item: Any) -> str:
        """Default key extraction - assumes item is string."""
        return str(item)
    
    def _get_prefix(self, item: Any) -> str:
        """Extract prefix from item using key_func."""
        key = self.key_func(item)
        return key[:self.prefix_length] if len(key) >= self.prefix_length else key
    
    def _compute_hash(self, item: Any) -> int:
        """Compute hash for item's prefix."""
        prefix = self._get_prefix(item)
        return self.hash_provider.hash(prefix)
    
    def _create_bucket(self, hash_key: int) -> Bucket:
        """Create new bucket with auto-incrementing ID."""
        bucket = Bucket(self._bucket_counter, hash_key)
        self._bucket_counter += 1
        self.buckets[hash_key] = bucket
        self._stats['num_buckets'] += 1
        return bucket
    
    def distribute(self, items: List[Any]) -> Dict[int, Bucket]:
        """
        Distribute items into buckets based on prefix hash.
        
        Args:
            items: List of items to distribute
        
        Returns:
            Dictionary mapping hash keys to buckets
        """
        for item in items:
            hash_key = self._compute_hash(item)
            
            if hash_key not in self.buckets:
                self._create_bucket(hash_key)
            
            self.buckets[hash_key].add(item)
            self._stats['total_items'] += 1
        
        self._update_stats()
        return self.buckets
    
    def add(self, item: Any) -> int:
        """
        Add single item to appropriate bucket.
        
        Args:
            item: Item to add
        
        Returns:
            Hash key of bucket where item was placed
        """
        hash_key = self._compute_hash(item)
        
        if hash_key not in self.buckets:
            self._create_bucket(hash_key)
        
        self.buckets[hash_key].add(item)
        self._stats['total_items'] += 1
        
        # Check if bucket needs splitting
        if self.max_bucket_size and len(self.buckets[hash_key]) > self.max_bucket_size:
            self._split_bucket(hash_key)
        
        return hash_key
    
    def _split_bucket(self, hash_key: int) -> None:
        """
        Split an oversized bucket by increasing prefix granularity.
        Creates sub-buckets with longer prefixes.
        """
        old_bucket = self.buckets[hash_key]
        old_prefix_len = self.prefix_length
        
        # Temporarily increase prefix length for this split
        temp_prefix_len = old_prefix_len + 1
        
        # Redistribute items with longer prefix
        temp_buckets: Dict[int, List[Any]] = defaultdict(list)
        
        for item in old_bucket.items:
            key = self.key_func(item)
            prefix = key[:temp_prefix_len] if len(key) >= temp_prefix_len else key
            new_hash = self.hash_provider.hash(prefix)
            temp_buckets[new_hash].append(item)
        
        # Replace old bucket with new ones
        del self.buckets[hash_key]
        self._stats['num_buckets'] -= 1
        
        for new_hash, items in temp_buckets.items():
            if new_hash not in self.buckets:
                self._create_bucket(new_hash)
            self.buckets[new_hash].add_batch(items)
    
    def get_bucket(self, hash_key: int) -> Optional[Bucket]:
        """Retrieve bucket by hash key."""
        return self.buckets.get(hash_key)
    
    def get_all_items(self, sorted_buckets: bool = False) -> List[Any]:
        """
        Retrieve all items from all buckets.
        
        Args:
            sorted_buckets: If True, sort each bucket before returning
        
        Returns:
            Flat list of all items
        """
        result = []
        
        # Sort buckets by hash key for deterministic ordering
        for hash_key in sorted(self.buckets.keys()):
            bucket = self.buckets[hash_key]
            if sorted_buckets:
                bucket.sort()
            result.extend(bucket.items)
        
        return result
    
    def merge_all(self, preserve_order: bool = False) -> Bucket:
        """
        Merge all buckets into a single bucket.
        
        Args:
            preserve_order: Maintain sorted order if buckets are sorted
        
        Returns:
            Single merged bucket
        """
        if not self.buckets:
            return Bucket(0, 0)
        
        # Start with first bucket
        sorted_hashes = sorted(self.buckets.keys())
        merged = self.buckets[sorted_hashes[0]]
        
        for hash_key in sorted_hashes[1:]:
            merged.merge(self.buckets[hash_key], preserve_order)
        
        return merged
    
    def iter_buckets(self) -> Iterator[Bucket]:
        """Iterate over all buckets in hash order."""
        for hash_key in sorted(self.buckets.keys()):
            yield self.buckets[hash_key]
    
    def _update_stats(self) -> None:
        """Update bucket statistics."""
        if not self.buckets:
            return
        
        sizes = [len(b) for b in self.buckets.values()]
        self._stats['largest_bucket'] = max(sizes)
        self._stats['smallest_bucket'] = min(sizes)
        self._stats['empty_buckets'] = sum(1 for s in sizes if s == 0)
    
    def get_stats(self) -> Dict[str, Any]:
        """Return current statistics."""
        self._update_stats()
        return self._stats.copy()
    
    def clear(self) -> None:
        """Clear all buckets and reset state."""
        self.buckets.clear()
        self._bucket_counter = 0
        self._stats = {
            'total_items': 0,
            'num_buckets': 0,
            'largest_bucket': 0,
            'smallest_bucket': float('inf'),
            'empty_buckets': 0
        }
    
    def __len__(self) -> int:
        return len(self.buckets)
    
    def __iter__(self) -> Iterator[Tuple[int, Bucket]]:
        """Iterate over (hash_key, bucket) pairs."""
        for hash_key in sorted(self.buckets.keys()):
            yield hash_key, self.buckets[hash_key]
    
    def __repr__(self) -> str:
        return f"BucketManager(buckets={len(self.buckets)}, items={self._stats['total_items']})"
