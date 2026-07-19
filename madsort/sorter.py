"""
Main sorting engine for madS0rt.
Orchestrates bucket creation, adaptive intra-bucket sorting, and k-way merge.
"""

from typing import List, Callable, Optional, Any, Dict, Union, Literal
import heapq
import bisect

from .bucket import Bucket, BucketManager
from .hash_utils import HashProvider, get_hash_provider


class SortStrategy:
    """
    Configuration for sorting strategy.
    Determines which algorithm to use based on bucket size.
    """
    
    # Thresholds for algorithm selection
    INSERTION_THRESHOLD = 10      # Use insertion sort for tiny buckets
    TIMSORT_THRESHOLD = 1000      # Python's default Timsort
    HYBRID_THRESHOLD = 10000      # Use bucket sort for very large buckets
    
    def __init__(
        self,
        insertion_threshold: int = 10,
        timsort_threshold: int = 1000,
        use_builtin_only: bool = False
    ):
        """
        Initialize sort strategy.
        
        Args:
            insertion_threshold: Max size for insertion sort
            timsort_threshold: Max size for Timsort (above this uses hybrid)
            use_builtin_only: Always use Python's sort (Timsort)
        """
        self.insertion_threshold = insertion_threshold
        self.timsort_threshold = timsort_threshold
        self.use_builtin_only = use_builtin_only
    
    def select_algorithm(self, bucket_size: int) -> Literal['insertion', 'timsort', 'hybrid']:
        """Select appropriate algorithm for bucket size."""
        if self.use_builtin_only:
            return 'timsort'
        
        if bucket_size <= self.insertion_threshold:
            return 'insertion'
        elif bucket_size <= self.timsort_threshold:
            return 'timsort'
        else:
            return 'hybrid'


class MadSorter:
    """
    Main sorting engine for madS0rt.
    Implements hybrid bucket-based sorting with adaptive strategies.
    """
    
    def __init__(
        self,
        prefix_length: int = 3,
        hash_provider: Optional[Union[str, HashProvider]] = None,
        key_func: Optional[Callable[[Any], Any]] = None,
        strategy: Optional[SortStrategy] = None,
        max_bucket_size: Optional[int] = None,
        copy_mode: bool = False
    ):
        """
        Initialize MadSorter.
        
        Args:
            prefix_length: Number of chars for prefix bucketing
            hash_provider: 'crc32', 'xxhash32', 'xxhash64', or HashProvider instance
            key_func: Function to extract sort key from items
            strategy: SortStrategy configuration (uses default if None)
            max_bucket_size: Auto-split buckets larger than this
            copy_mode: If True, don't modify original list (return new sorted list)
        """
        self.prefix_length = prefix_length
        self.key_func = key_func or (lambda x: x)
        self.strategy = strategy or SortStrategy()
        self.max_bucket_size = max_bucket_size
        self.copy_mode = copy_mode
        
        # Initialize hash provider
        if isinstance(hash_provider, str):
            self.hash_provider = get_hash_provider(hash_provider)
        elif hash_provider is None:
            self.hash_provider = get_hash_provider("crc32")
        else:
            self.hash_provider = hash_provider
        
        self._bucket_manager: Optional[BucketManager] = None
        self._stats = {
            'total_items': 0,
            'num_buckets': 0,
            'sort_time_ms': 0,
            'merge_time_ms': 0,
            'total_time_ms': 0,
        }
    
    def _create_bucket_manager(self) -> BucketManager:
        """Create configured BucketManager."""
        return BucketManager(
            hash_provider=self.hash_provider,
            prefix_length=self.prefix_length,
            key_func=self._get_string_key,
            max_bucket_size=self.max_bucket_size
        )
    
    def _get_string_key(self, item: Any) -> str:
        """Convert item to string key for bucketing."""
        key = self.key_func(item)
        return str(key) if key is not None else ""
    
    def _adaptive_sort_bucket(self, bucket: Bucket, reverse: bool = False) -> None:
        """
        Sort bucket using adaptive strategy based on size.
        
        Args:
            bucket: Bucket to sort
            reverse: Sort in descending order
        """
        size = len(bucket)
        algorithm = self.strategy.select_algorithm(size)
        
        if algorithm == 'insertion':
            self._insertion_sort(bucket, reverse)
        else:
            # Use Python's Timsort (highly optimized)
            bucket.sort(reverse=reverse, key=self.key_func)
        
        # Mark as sorted
        bucket._sorted = True
    
    def _insertion_sort(self, bucket: Bucket, reverse: bool = False) -> None:
        """
        Manual insertion sort for very small buckets.
        More efficient than Timsort overhead for tiny lists.
        """
        items = bucket.items
        key_func = self.key_func
        
        for i in range(1, len(items)):
            key_item = items[i]
            key_val = key_func(key_item)
            j = i - 1
            
            while j >= 0:
                current_val = key_func(items[j])
                
                if reverse:
                    should_move = current_val < key_val
                else:
                    should_move = current_val > key_val
                
                if not should_move:
                    break
                
                items[j + 1] = items[j]
                j -= 1
            
            items[j + 1] = key_item
    
    def _k_way_merge(self, buckets: List[Bucket], reverse: bool = False) -> List[Any]:
        """
        Merge multiple sorted buckets using k-way merge with heap.
        More efficient than sequential merging for many buckets.
        
        Args:
            buckets: List of sorted buckets
            reverse: Merge in descending order
        
        Returns:
            Merged sorted list
        """
        if not buckets:
            return []
        
        if len(buckets) == 1:
            return list(buckets[0].items)
        
        # Use heapq.merge for efficient k-way merge
        key_func = self.key_func
        
        # Create iterators for each bucket
        iterators = [iter(bucket.items) for bucket in buckets]
        
        # Use heapq.merge which implements efficient k-way merge
        # It yields items in sorted order without loading everything into memory
        merged = heapq.merge(
            *iterators,
            key=key_func,
            reverse=reverse
        )
        
        return list(merged)
    
    def _sequential_merge(self, buckets: List[Bucket], reverse: bool = False) -> List[Any]:
        """
        Alternative: Sequential merge (simpler, less memory overhead for few buckets).
        
        Args:
            buckets: List of sorted buckets
            reverse: Merge in descending order
        
        Returns:
            Merged sorted list
        """
        result = []
        key_func = self.key_func
        
        for bucket in buckets:
            items = list(bucket.items)
            if reverse:
                items.reverse()
            
            # Merge into result maintaining sorted order
            for item in items:
                # Find insertion point using binary search
                key_val = key_func(item)
                pos = bisect.bisect_right([key_func(x) for x in result], key_val)
                result.insert(pos, item)
        
        return result
    
    def sort(self, items: List[Any], reverse: bool = False) -> List[Any]:
        """
        Sort items using madS0rt hybrid algorithm.
        
        Algorithm:
        1. Distribute items into prefix-based buckets
        2. Sort each bucket with adaptive strategy
        3. Merge sorted buckets using k-way merge
        
        Args:
            items: List of items to sort
            reverse: Sort in descending order if True
        
        Returns:
            New sorted list (or None if in-place mode)
        """
        import time
        
        if not items:
            return [] if self.copy_mode else items
        
        start_time = time.perf_counter()
        
        # Phase 1: Create buckets
        self._bucket_manager = self._create_bucket_manager()
        
        if self.copy_mode:
            items_to_sort = items.copy()
        else:
            items_to_sort = items
        
        self._bucket_manager.distribute(items_to_sort)
        
        # Phase 2: Sort each bucket
        sort_start = time.perf_counter()
        
        buckets = list(self._bucket_manager.iter_buckets())
        bucket_list = [b for _, b in buckets]
        
        for bucket in bucket_list:
            self._adaptive_sort_bucket(bucket, reverse)
        
        sort_end = time.perf_counter()
        self._stats['sort_time_ms'] = (sort_end - sort_start) * 1000
        
        # Phase 3: Merge buckets
        merge_start = time.perf_counter()
        
        # Use k-way merge for efficiency
        result = self._k_way_merge(bucket_list, reverse)
        
        merge_end = time.perf_counter()
        self._stats['merge_time_ms'] = (merge_end - merge_start) * 1000
        
        # Update stats
        end_time = time.perf_counter()
        self._stats['total_time_ms'] = (end_time - start_time) * 1000
        self._stats['total_items'] = len(items)
        self._stats['num_buckets'] = len(bucket_list)
        
        # Replace original if not copy mode
        if not self.copy_mode:
            items.clear()
            items.extend(result)
            return items
        
        return result
    
    def sorted(self, items: List[Any], reverse: bool = False) -> List[Any]:
        """
        Return new sorted list (like built-in sorted()).
        Always returns new list, never modifies original.
        
        Args:
            items: List of items to sort
            reverse: Sort in descending order
        
        Returns:
            New sorted list
        """
        original_mode = self.copy_mode
        self.copy_mode = True
        try:
            return self.sort(items, reverse)
        finally:
            self.copy_mode = original_mode
    
    def get_stats(self) -> Dict[str, Any]:
        """Get sorting statistics."""
        if self._bucket_manager:
            bucket_stats = self._bucket_manager.get_stats()
        else:
            bucket_stats = {}
        
        return {
            **self._stats,
            'bucket_stats': bucket_stats,
        }
    
    def get_buckets(self) -> Dict[int, Bucket]:
        """Get bucket distribution (for inspection/debugging)."""
        if self._bucket_manager:
            return self._bucket_manager.buckets
        return {}
    
    def reset(self) -> None:
        """Reset sorter state."""
        self._bucket_manager = None
        self._stats = {
            'total_items': 0,
            'num_buckets': 0,
            'sort_time_ms': 0,
            'merge_time_ms': 0,
            'total_time_ms': 0,
        }


# Convenience functions (similar to built-in sorted/sort)
def madsort(
    items: List[Any],
    key: Optional[Callable] = None,
    reverse: bool = False,
    prefix_length: int = 3,
    copy: bool = True
) -> List[Any]:
    """
    Convenience function for one-time madS0rt sorting.
    
    Args:
        items: List to sort
        key: Key extraction function
        reverse: Descending order
        prefix_length: Prefix length for bucketing
        copy: If True, return new list; if False, sort in-place
    
    Returns:
        Sorted list
    """
    sorter = MadSorter(
        prefix_length=prefix_length,
        key_func=key,
        copy_mode=copy
    )
    return sorter.sort(items, reverse)


def madsorted(
    items: List[Any],
    key: Optional[Callable] = None,
    reverse: bool = False,
    prefix_length: int = 3
) -> List[Any]:
    """
    Like built-in sorted() - always returns new sorted list.
    
    Args:
        items: List to sort
        key: Key extraction function
        reverse: Descending order
        prefix_length: Prefix length for bucketing
    
    Returns:
        New sorted list
    """
    return madsort(items, key, reverse, prefix_length, copy=True)
