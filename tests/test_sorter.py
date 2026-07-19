"""
Tests for madS0rt main sorting engine.
"""

import pytest
import random
import string
from madsort.sorter import MadSorter, SortStrategy, madsort, madsorted


class TestSortStrategy:
    """Test SortStrategy configuration."""
    
    def test_default_thresholds(self):
        strategy = SortStrategy()
        assert strategy.select_algorithm(5) == 'insertion'
        assert strategy.select_algorithm(100) == 'timsort'
        assert strategy.select_algorithm(5000) == 'hybrid'
    
    def test_custom_thresholds(self):
        strategy = SortStrategy(
            insertion_threshold=20,
            timsort_threshold=500
        )
        assert strategy.select_algorithm(15) == 'insertion'
        assert strategy.select_algorithm(300) == 'timsort'
        assert strategy.select_algorithm(1000) == 'hybrid'
    
    def test_builtin_only(self):
        strategy = SortStrategy(use_builtin_only=True)
        assert strategy.select_algorithm(5) == 'timsort'
        assert strategy.select_algorithm(10000) == 'timsort'


class TestMadSorterBasic:
    """Test basic MadSorter functionality."""
    
    def test_empty_list(self):
        sorter = MadSorter()
        result = sorter.sort([])
        assert result == []
    
    def test_single_item(self):
        sorter = MadSorter()
        result = sorter.sort([42])
        assert result == [42]
    
    def test_already_sorted(self):
        sorter = MadSorter()
        items = [1, 2, 3, 4, 5]
        result = sorter.sort(items.copy())
        assert result == [1, 2, 3, 4, 5]
    
    def test_reverse_sorted(self):
        sorter = MadSorter()
        items = [5, 4, 3, 2, 1]
        result = sorter.sort(items.copy())
        assert result == [1, 2, 3, 4, 5]
    
    def test_reverse_order(self):
        sorter = MadSorter()
        items = [1, 3, 2, 5, 4]
        result = sorter.sort(items.copy(), reverse=True)
        assert result == [5, 4, 3, 2, 1]
    
    def test_with_key_function(self):
        sorter = MadSorter(key_func=len)
        items = ["aaa", "bb", "c", "dddd"]
        result = sorter.sort(items.copy())
        assert result == ["c", "bb", "aaa", "dddd"]
    
    def test_copy_mode(self):
        sorter = MadSorter(copy_mode=True)
        original = [3, 1, 2]
        result = sorter.sort(original)
        assert result == [1, 2, 3]
        assert original == [3, 1, 2]
    
    def test_inplace_mode(self):
        sorter = MadSorter(copy_mode=False)
        original = [3, 1, 2]
        result = sorter.sort(original)
        assert result == [1, 2, 3]
        assert original == [1, 2, 3]


class TestMadSorterAlgorithms:
    """Test different sorting algorithms."""
    
    def test_insertion_sort_small(self):
        strategy = SortStrategy(insertion_threshold=10)
        sorter = MadSorter(strategy=strategy)
        
        items = [5, 2, 8, 1, 9, 3]
        result = sorter.sort(items.copy())
        assert result == [1, 2, 3, 5, 8, 9]
    
    def test_large_dataset(self):
        sorter = MadSorter(prefix_length=2)
        
        items = list(range(1000, 0, -1))
        result = sorter.sort(items.copy())
        assert result == list(range(1, 1001))
    
    def test_string_sorting(self):
        sorter = MadSorter(prefix_length=2)
        
        items = ["zebra", "apple", "banana", "cherry", "date"]
        result = sorter.sort(items.copy())
        assert result == ["apple", "banana", "cherry", "date", "zebra"]


class TestKWayMerge:
    """Test k-way merge functionality."""
    
    def test_merge_two_buckets(self):
        sorter = MadSorter(prefix_length=1)
        
        items = ["banana", "apple", "apricot", "blueberry"]
        result = sorter.sort(items.copy())
        assert result == ["apple", "apricot", "banana", "blueberry"]
    
    def test_merge_multiple_buckets(self):
        sorter = MadSorter(prefix_length=1)
        
        items = ["zebra", "apple", "banana", "cherry", "date", "elderberry"]
        result = sorter.sort(items.copy())
        assert result == ["apple", "banana", "cherry", "date", "elderberry", "zebra"]


class TestConvenienceFunctions:
    """Test madsort and madsorted functions."""
    
    def test_madsort_copies(self):
        original = [3, 1, 2]
        result = madsort(original)
        assert result == [1, 2, 3]
        assert original == [3, 1, 2]
    
    def test_madsorted(self):
        original = [3, 1, 2]
        result = madsorted(original)
        assert result == [1, 2, 3]
        assert original == [3, 1, 2]


class TestComplexScenarios:
    """Test complex real-world scenarios."""
    
    def test_with_custom_extractor(self):
        from madsort.extractors import NumericExtractor
        
        extractor = NumericExtractor(default=0)
        
        items = ["item_10", "item_2", "item_1", "item_20"]
        sorter = MadSorter(key_func=extractor)
        result = sorter.sort(items.copy())
        assert result == ["item_1", "item_2", "item_10", "item_20"]
    
    def test_stability(self):
        """Test that sort is stable."""
        sorter = MadSorter(key_func=lambda x: x[0])
        
        items = [('b', 1), ('a', 2), ('b', 3), ('a', 4)]
        result = sorter.sort(items.copy())
        
        assert result == [('a', 2), ('a', 4), ('b', 1), ('b', 3)]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
