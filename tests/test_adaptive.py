"""
Tests for madS0rt adaptive bucketing system.
"""

import pytest
import random
import string
from madsort.adaptive import (
    DistributionAnalyzer,
    LoadBalancer,
    AdaptiveBucketManager,
    AdaptiveMadSorter,
)
from madsort.bucket import BucketManager


class TestDistributionAnalyzer:
    """Test DistributionAnalyzer."""
    
    def test_uniform_distribution(self):
        analyzer = DistributionAnalyzer()
        
        items = ['a' + str(i) for i in range(100)] + \
                ['b' + str(i) for i in range(100)] + \
                ['c' + str(i) for i in range(100)]
        
        stats = analyzer.analyze(items, lambda x: x)
        
        assert stats['distribution_type'] in ['uniform', 'dense']
        assert stats['optimal_prefix_length'] >= 1
    
    def test_sparse_distribution(self):
        analyzer = DistributionAnalyzer()
        items = ['zzz' + str(i) for i in range(100)]
        
        stats = analyzer.analyze(items, lambda x: x)
        assert stats['distribution_type'] in ['sparse', 'skewed', 'dense']
    
    def test_empty_list(self):
        analyzer = DistributionAnalyzer()
        stats = analyzer.analyze([], lambda x: x)
        assert stats['distribution_type'] == 'unknown'
    
    def test_entropy_calculation(self):
        analyzer = DistributionAnalyzer()
        items = ['a', 'b', 'c', 'd'] * 25
        stats = analyzer.analyze(items, lambda x: x, prefix_lengths=[1])
        
        entropy = stats['analysis_by_length'][1]['entropy']
        assert entropy > 0.9
    
    def test_recommendations(self):
        analyzer = DistributionAnalyzer()
        items = ['a' * i for i in range(1, 101)]
        analyzer.analyze(items, lambda x: x)
        recs = analyzer.get_recommendations()
        
        assert len(recs) > 0


class TestLoadBalancer:
    """Test LoadBalancer."""
    
    def test_balance_even_buckets(self):
        manager = BucketManager(prefix_length=1)
        items = ['a' + str(i) for i in range(50)] + \
                ['b' + str(i) for i in range(50)]
        
        manager.distribute(items)
        balancer = LoadBalancer(target_size=50)
        balancer.balance(manager)
        
        assert len(balancer.migrations) == 0
    
    def test_migration_report(self):
        balancer = LoadBalancer()
        report = balancer.get_migration_report()
        assert "No migrations" in report


class TestAdaptiveBucketManager:
    """Test AdaptiveBucketManager."""
    
    def test_auto_adjust_enabled(self):
        manager = AdaptiveBucketManager(
            prefix_length=1,
            auto_adjust=True,
            enable_load_balance=False
        )
        
        items = ['a' + str(i) for i in range(100)]
        manager.distribute(items)
        
        stats = manager.get_adaptive_stats()
        assert stats['current_prefix_length'] >= 1
    
    def test_force_rebalance(self):
        manager = AdaptiveBucketManager(prefix_length=1)
        items = ['a' + str(i) for i in range(100)]
        manager.distribute(items)
        
        initial_count = manager.get_adaptive_stats()['load_balances_performed']
        manager.force_rebalance()
        final_count = manager.get_adaptive_stats()['load_balances_performed']
        assert final_count >= initial_count


class TestAdaptiveMadSorter:
    """Test AdaptiveMadSorter."""
    
    def test_basic_sort(self):
        sorter = AdaptiveMadSorter(
            initial_prefix_length=2,
            auto_adjust=False,
            enable_load_balance=False
        )
        
        items = [3, 1, 4, 1, 5, 9, 2, 6]
        result = sorter.sort(items)
        
        assert result == [1, 1, 2, 3, 4, 5, 6, 9]
    
    def test_adaptive_report(self):
        sorter = AdaptiveMadSorter()
        items = [3, 1, 4, 1, 5]
        sorter.sort(items)
        
        report = sorter.get_adaptive_report()
        assert "Adaptive Sorting Report" in report
    
    def test_empty_list(self):
        sorter = AdaptiveMadSorter()
        result = sorter.sort([])
        assert result == []


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
