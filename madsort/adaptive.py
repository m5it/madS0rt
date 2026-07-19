"""
Adaptive bucketing system for madS0rt.
Auto-adjusts bucket depth based on data distribution and implements load balancing.
"""

from typing import List, Dict, Any, Optional, Callable, Tuple
from collections import defaultdict
import math

from .bucket import Bucket, BucketManager
from .hash_utils import HashProvider, get_hash_provider


class DistributionAnalyzer:
    """
    Analyzes data distribution to determine optimal bucketing strategy.
    """
    
    def __init__(self):
        self.stats = {
            'total_items': 0,
            'unique_prefixes': 0,
            'max_bucket_size': 0,
            'min_bucket_size': float('inf'),
            'avg_bucket_size': 0.0,
            'std_deviation': 0.0,
            'entropy': 0.0,
            'distribution_type': 'unknown',
        }
    
    def analyze(self, items: List[Any], key_func: Callable[[Any], str], 
                prefix_lengths: List[int] = [1, 2, 3, 4]) -> Dict[str, Any]:
        """
        Analyze data distribution across different prefix lengths.
        
        Args:
            items: Items to analyze
            key_func: Function to extract string keys
            prefix_lengths: Prefix lengths to test
        
        Returns:
            Analysis results with recommendations
        """
        if not items:
            return self.stats
        
        results = {}
        
        for length in prefix_lengths:
            prefix_counts = defaultdict(int)
            
            for item in items:
                key = key_func(item)
                prefix = key[:length] if len(key) >= length else key
                prefix_counts[prefix] += 1
            
            counts = list(prefix_counts.values())
            n = len(items)
            k = len(counts)
            
            results[length] = {
                'num_buckets': k,
                'max_size': max(counts) if counts else 0,
                'min_size': min(counts) if counts else 0,
                'avg_size': n / k if k > 0 else 0,
                'std_dev': self._std_dev(counts),
                'entropy': self._entropy(counts, n),
                'load_factor': n / k if k > 0 else 0,
            }
        
        optimal = self._find_optimal(results)
        
        self.stats = {
            'total_items': len(items),
            'analysis_by_length': results,
            'optimal_prefix_length': optimal['length'],
            'optimal_stats': optimal['stats'],
            'distribution_type': self._classify_distribution(optimal['stats']),
        }
        
        return self.stats
    
    def _std_dev(self, values: List[int]) -> float:
        """Calculate standard deviation."""
        if not values:
            return 0.0
        
        mean = sum(values) / len(values)
        variance = sum((x - mean) ** 2 for x in values) / len(values)
        return math.sqrt(variance)
    
    def _entropy(self, counts: List[int], total: int) -> float:
        """Calculate Shannon entropy."""
        if total == 0:
            return 0.0
        
        entropy = 0.0
        for count in counts:
            if count > 0:
                p = count / total
                entropy -= p * math.log2(p)
        
        max_entropy = math.log2(len(counts)) if len(counts) > 1 else 1
        return entropy / max_entropy if max_entropy > 0 else 0.0
    
    def _find_optimal(self, results: Dict[int, Dict]) -> Dict[str, Any]:
        """Find optimal prefix length."""
        best_score = -1
        optimal = None
        
        for length, stats in results.items():
            entropy_score = stats['entropy']
            
            if stats['avg_size'] > 0:
                balance_score = 1.0 - min(stats['std_dev'] / stats['avg_size'], 1.0)
            else:
                balance_score = 0.0
            
            if 10 <= stats['num_buckets'] <= 1000:
                size_score = 1.0
            elif stats['num_buckets'] < 10:
                size_score = stats['num_buckets'] / 10
            else:
                size_score = max(0, 1.0 - (stats['num_buckets'] - 1000) / 9000)
            
            total_score = (entropy_score * 0.4 + balance_score * 0.4 + size_score * 0.2)
            
            if total_score > best_score:
                best_score = total_score
                optimal = {'length': length, 'stats': stats, 'score': total_score}
        
        return optimal or {'length': 3, 'stats': {}, 'score': 0}
    
    def _classify_distribution(self, stats: Dict[str, Any]) -> str:
        """Classify distribution type."""
        if not stats:
            return 'unknown'
        
        entropy = stats.get('entropy', 0)
        std_dev = stats.get('std_dev', 0)
        avg_size = stats.get('avg_size', 1)
        
        cv = std_dev / avg_size if avg_size > 0 else 0
        
        if entropy > 0.8 and cv < 0.5:
            return 'uniform'
        elif entropy < 0.3:
            return 'sparse'
        elif cv > 1.0:
            return 'skewed'
        else:
            return 'dense'
    
    def get_recommendations(self) -> List[str]:
        """Get recommendations based on analysis."""
        if self.stats['distribution_type'] == 'unknown':
            return ['No data to analyze']
        
        recs = []
        dist_type = self.stats['distribution_type']
        optimal_length = self.stats.get('optimal_prefix_length', 3)
        
        if dist_type == 'sparse':
            recs.append(f'Data is sparse, reduce prefix length to {max(1, optimal_length - 1)}')
        elif dist_type == 'dense':
            recs.append(f'Data is dense, increase prefix length to {optimal_length + 1}')
        elif dist_type == 'skewed':
            recs.append('Data is skewed, use load balancing')
        elif dist_type == 'uniform':
            recs.append('Data is well distributed')
        
        recs.append(f'Recommended prefix length: {optimal_length}')
        return recs


class LoadBalancer:
    """
    Balances load across buckets to handle uneven distributions.
    """
    
    def __init__(self, target_size: Optional[int] = None, 
                 max_imbalance: float = 2.0):
        self.target_size = target_size
        self.max_imbalance = max_imbalance
        self.migrations = []
    
    def balance(self, bucket_manager: BucketManager) -> Dict[int, Bucket]:
        """Rebalance buckets."""
        if not bucket_manager.buckets:
            return {}
        
        buckets = bucket_manager.buckets
        sizes = {k: len(b) for k, b in buckets.items()}
        
        if not self.target_size:
            avg_size = sum(sizes.values()) / len(sizes)
            self.target_size = int(avg_size)
        
        overloaded = {k: s for k, s in sizes.items() 
                     if s > self.target_size * self.max_imbalance}
        underloaded = {k: s for k, s in sizes.items() 
                      if s < self.target_size / self.max_imbalance}
        
        for over_hash, over_size in sorted(overloaded.items(), key=lambda x: -x[1]):
            over_bucket = buckets[over_hash]
            
            for under_hash in sorted(underloaded.keys()):
                if over_size <= self.target_size:
                    break
                
                under_bucket = buckets[under_hash]
                space_available = self.target_size - sizes[under_hash]
                items_to_move = min(space_available, over_size - self.target_size)
                
                items = over_bucket.items[:items_to_move]
                over_bucket.items = over_bucket.items[items_to_move:]
                under_bucket.add_batch(items)
                
                sizes[over_hash] -= items_to_move
                sizes[under_hash] += items_to_move
                self.migrations.append((over_hash, under_hash, items_to_move))
        
        return buckets
    
    def get_migration_report(self) -> str:
        """Get report of migrations."""
        if not self.migrations:
            return "No migrations performed"
        
        lines = ["Load Balancing Report:", "-" * 40]
        total_moved = sum(count for _, _, count in self.migrations)
        
        for from_hash, to_hash, count in self.migrations:
            lines.append(f"  Moved {count} items from {from_hash:08x} to {to_hash:08x}")
        
        lines.append(f"Total items migrated: {total_moved}")
        return "\n".join(lines)


class AdaptiveBucketManager(BucketManager):
    """
    Extended BucketManager with adaptive capabilities.
    """
    
    def __init__(
        self,
        hash_provider = None,
        prefix_length: int = 3,
        key_func = None,
        max_bucket_size: Optional[int] = None,
        auto_adjust: bool = True,
        enable_load_balance: bool = True
    ):
        super().__init__(hash_provider, prefix_length, key_func, max_bucket_size)
        
        self.auto_adjust = auto_adjust
        self.enable_load_balance = enable_load_balance
        self.analyzer = DistributionAnalyzer()
        self.load_balancer = LoadBalancer()
        self._adaptive_stats = {
            'adjustments_made': 0,
            'load_balances_performed': 0,
            'optimal_lengths_tried': [],
        }
    
    def distribute(self, items: List[Any]) -> Dict[int, Bucket]:
        """Distribute items with adaptive optimization."""
        if not items:
            return self.buckets
        
        if self.auto_adjust:
            analysis = self.analyzer.analyze(items, self.key_func)
            optimal_length = analysis.get('optimal_prefix_length', self.prefix_length)
            
            if optimal_length != self.prefix_length:
                self.prefix_length = optimal_length
                self._adaptive_stats['adjustments_made'] += 1
                self._adaptive_stats['optimal_lengths_tried'].append(optimal_length)
        
        super().distribute(items)
        
        if self.enable_load_balance:
            self.load_balancer.balance(self)
            self._adaptive_stats['load_balances_performed'] += 1
        
        return self.buckets
    
    def get_adaptive_stats(self) -> Dict[str, Any]:
        """Get adaptive bucketing statistics."""
        return {
            **self._adaptive_stats,
            'current_prefix_length': self.prefix_length,
            'distribution_analysis': self.analyzer.stats if self.auto_adjust else None,
            'last_migration_report': self.load_balancer.get_migration_report() 
                                    if self.enable_load_balance else None,
        }
    
    def force_rebalance(self) -> Dict[int, Bucket]:
        """Force immediate load rebalancing."""
        self.load_balancer.balance(self)
        self._adaptive_stats['load_balances_performed'] += 1
        return self.buckets


class AdaptiveMadSorter:
    """
    MadSorter with full adaptive capabilities.
    """
    
    def __init__(
        self,
        initial_prefix_length: int = 3,
        hash_provider = None,
        key_func = None,
        auto_adjust: bool = True,
        enable_load_balance: bool = True,
        max_bucket_size: Optional[int] = None
    ):
        self.initial_prefix_length = initial_prefix_length
        self.key_func = key_func or (lambda x: x)
        self.auto_adjust = auto_adjust
        self.enable_load_balance = enable_load_balance
        self.max_bucket_size = max_bucket_size
        
        if isinstance(hash_provider, str):
            self.hash_provider = get_hash_provider(hash_provider)
        elif hash_provider is None:
            self.hash_provider = get_hash_provider("crc32")
        else:
            self.hash_provider = hash_provider
        
        self._analyzer = DistributionAnalyzer()
        self._sorter = None
        self._adaptive_history = []
    
    def sort(self, items: List[Any], reverse: bool = False) -> List[Any]:
        """Sort with adaptive optimization."""
        if not items:
            return []
        
        prefix_length = self.initial_prefix_length
        
        if self.auto_adjust and len(items) > 100:
            analysis = self._analyzer.analyze(items, 
                lambda x: str(self.key_func(x)))
            prefix_length = analysis.get('optimal_prefix_length', 
                                        self.initial_prefix_length)
        
        from .sorter import MadSorter
        
        self._sorter = MadSorter(
            prefix_length=prefix_length,
            hash_provider=self.hash_provider,
            key_func=self.key_func,
            max_bucket_size=self.max_bucket_size,
            copy_mode=True
        )
        
        if self.enable_load_balance:
            self._sorter._bucket_manager = AdaptiveBucketManager(
                hash_provider=self.hash_provider,
                prefix_length=prefix_length,
                key_func=lambda x: str(self.key_func(x)),
                max_bucket_size=self.max_bucket_size,
                auto_adjust=self.auto_adjust,
                enable_load_balance=True
            )
        
        result = self._sorter.sort(items, reverse)
        
        self._adaptive_history.append({
            'items_count': len(items),
            'prefix_length_used': prefix_length,
            'analysis': self._analyzer.stats if self.auto_adjust else None,
        })
        
        return result
    
    def get_adaptive_report(self) -> str:
        """Get report of adaptive optimizations."""
        lines = ["Adaptive Sorting Report:", "=" * 50]
        
        for i, record in enumerate(self._adaptive_history, 1):
            lines.append(f"\nSort #{i}:")
            lines.append(f"  Items: {record['items_count']}")
            lines.append(f"  Prefix length: {record['prefix_length_used']}")
            
            if record['analysis']:
                lines.append(f"  Distribution: {record['analysis'].get('distribution_type', 'unknown')}")
        
        return "\n".join(lines)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive statistics."""
        if self._sorter:
            base_stats = self._sorter.get_stats()
        else:
            base_stats = {}
        
        return {
            **base_stats,
            'adaptive_history': self._adaptive_history,
            'auto_adjust_enabled': self.auto_adjust,
            'load_balance_enabled': self.enable_load_balance,
        }
