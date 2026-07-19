#!/usr/bin/env python3
"""
Basic usage examples for madS0rt.
"""

from madsort import MadSorter, madsorted, madsort


def example_1_simple_sorting():
    """Example 1: Simple string sorting."""
    print("Example 1: Simple String Sorting")
    print("-" * 40)
    
    words = ["zebra", "apple", "banana", "cherry", "date"]
    result = madsorted(words)
    
    print(f"Input:  {words}")
    print(f"Output: {result}")
    print()


def example_2_custom_key():
    """Example 2: Sorting with custom key function."""
    print("Example 2: Custom Key Function")
    print("-" * 40)
    
    # Sort by string length
    words = ["aaa", "bb", "c", "dddd", "eeeee"]
    result = madsorted(words, key=len)
    
    print(f"Input:  {words}")
    print(f"Output: {result} (sorted by length)")
    print()


def example_3_numeric_sorting():
    """Example 3: Proper numeric sorting."""
    print("Example 3: Numeric Sorting")
    print("-" * 40)
    
    from madsort import NumericExtractor
    
    items = ["item_10", "item_2", "item_1", "item_20"]
    
    # Without numeric extraction: lexicographic
    result_lex = madsorted(items)
    print(f"Lexicographic: {result_lex}")
    
    # With numeric extraction
    extractor = NumericExtractor()
    result_num = madsorted(items, key=extractor)
    print(f"Numeric:       {result_num}")
    print()


def example_4_prefix_bucketing():
    """Example 4: Understanding prefix bucketing."""
    print("Example 4: Prefix Bucketing")
    print("-" * 40)
    
    sorter = MadSorter(prefix_length=2)
    
    words = ["apple", "apricot", "banana", "blueberry", "cherry"]
    sorter._bucket_manager.distribute(words)
    
    print("Bucket distribution:")
    for hash_key, bucket in sorter._bucket_manager:
        print(f"  Bucket {hash_key:08x}: {list(bucket)}")
    print()


def example_5_inplace_vs_copy():
    """Example 5: In-place vs copy mode."""
    print("Example 5: In-place vs Copy Mode")
    print("-" * 40)
    
    # Copy mode (default) - preserves original
    original = [3, 1, 4, 1, 5]
    result = madsort(original, copy=True)
    print(f"Original preserved: {original}")
    print(f"New sorted:         {result}")
    
    # In-place mode - modifies original
    original = [3, 1, 4, 1, 5]
    result = madsort(original, copy=False)
    print(f"Modified original:  {original}")
    print(f"Returned:           {result}")
    print()


def example_6_statistics():
    """Example 6: Getting performance statistics."""
    print("Example 6: Performance Statistics")
    print("-" * 40)
    
    sorter = MadSorter(prefix_length=3)
    result = sorter.sort([3, 1, 4, 1, 5, 9, 2, 6])
    
    stats = sorter.get_stats()
    print(f"Sorted {stats['total_items']} items")
    print(f"  Buckets:     {stats['num_buckets']}")
    print(f"  Sort time:   {stats['sort_time_ms']:.3f}ms")
    print(f"  Merge time:  {stats['merge_time_ms']:.3f}ms")
    print(f"  Total time:   {stats['total_time_ms']:.3f}ms")
    print()


if __name__ == "__main__":
    example_1_simple_sorting()
    example_2_custom_key()
    example_3_numeric_sorting()
    example_4_prefix_bucketing()
    example_5_inplace_vs_copy()
    example_6_statistics()
