"""
Benchmark suite for madS0rt.
Compares performance against Python's built-in sorted() and Timsort.
"""

import time
import random
import string
import sys
from typing import List, Callable, Dict, Any, Optional
from dataclasses import dataclass
import statistics


@dataclass
class BenchmarkResult:
    """Result of a single benchmark run."""
    name: str
    items_count: int
    times_ms: List[float]
    
    @property
    def avg_time_ms(self) -> float:
        return statistics.mean(self.times_ms)
    
    @property
    def min_time_ms(self) -> float:
        return min(self.times_ms)
    
    @property
    def max_time_ms(self) -> float:
        return max(self.times_ms)


class BenchmarkSuite:
    """Comprehensive benchmark suite for sorting algorithms."""
    
    def __init__(self, iterations: int = 3, warmup: int = 1):
        self.iterations = iterations
        self.warmup = warmup
        self.results: List[BenchmarkResult] = []
    
    def generate_data(self, size: int, data_type: str = 'random_strings') -> List[Any]:
        """Generate test data."""
        if data_type == 'random_strings':
            return [''.join(random.choices(string.ascii_lowercase, k=10)) 
                    for _ in range(size)]
        elif data_type == 'sorted':
            return [f"item_{i:08d}" for i in range(size)]
        elif data_type == 'reverse':
            return [f"item_{i:08d}" for i in range(size, 0, -1)]
        elif data_type == 'few_unique':
            choices = ['alpha', 'beta', 'gamma', 'delta']
            return [random.choice(choices) + str(i) for i in range(size)]
        elif data_type == 'realistic':
            words = ['apple', 'banana', 'cherry', 'date', 'elderberry', 'fig', 'grape']
            return [random.choice(words) + '_' + ''.join(random.choices(string.digits, k=4))
                    for _ in range(size)]
        else:
            return list(range(size))
    
    def time_sort(self, sort_func: Callable, data: List[Any]) -> float:
        """Time a sorting function."""
        for _ in range(self.warmup):
            sort_func(data.copy())
        
        times = []
        for _ in range(self.iterations):
            data_copy = data.copy()
            start = time.perf_counter()
            sort_func(data_copy)
            end = time.perf_counter()
            times.append((end - start) * 1000)
        
        return statistics.median(times)
    
    def benchmark_python_sorted(self, data: List[Any]) -> BenchmarkResult:
        """Benchmark Python's built-in sorted()."""
        sort_func = lambda x: sorted(x)
        times = [self.time_sort(sort_func, data) for _ in range(self.iterations)]
        
        return BenchmarkResult(
            name='Python sorted()',
            items_count=len(data),
            times_ms=times
        )
    
    def benchmark_list_sort(self, data: List[Any]) -> BenchmarkResult:
        """Benchmark list.sort() (Timsort in-place)."""
        def sort_func(x):
            x_copy = x.copy()
            x_copy.sort()
            return x_copy
        
        times = [self.time_sort(sort_func, data) for _ in range(self.iterations)]
        
        return BenchmarkResult(
            name='list.sort() (Timsort)',
            items_count=len(data),
            times_ms=times
        )
    
    def benchmark_mads0rt(self, data: List[Any], prefix_length: int = 3, adaptive: bool = False) -> BenchmarkResult:
        """Benchmark madS0rt."""
        if adaptive:
            from .adaptive import AdaptiveMadSorter
            sorter = AdaptiveMadSorter(
                initial_prefix_length=prefix_length,
                auto_adjust=True,
                enable_load_balance=True
            )
            sort_func = lambda x: sorter.sort(x.copy())
            name = 'madS0rt (adaptive)'
        else:
            from .sorter import MadSorter
            sorter = MadSorter(prefix_length=prefix_length, copy_mode=True)
            sort_func = lambda x: sorter.sort(x.copy())
            name = f'madS0rt (prefix={prefix_length})'
        
        times = [self.time_sort(sort_func, data) for _ in range(self.iterations)]
        
        return BenchmarkResult(
            name=name,
            items_count=len(data),
            times_ms=times
        )
    
    def print_results(self, results: List[BenchmarkResult]) -> None:
        """Print formatted results."""
        print(f"\n{'Algorithm':<25} {'Items':>10} {'Avg (ms)':>12} {'Min (ms)':>12} {'Max (ms)':>12}")
        print("-" * 75)
        
        for r in results:
            print(f"{r.name:<25} {r.items_count:>10,} {r.avg_time_ms:>12.2f} "
                  f"{r.min_time_ms:>12.2f} {r.max_time_ms:>12.2f}")


def run_quick_benchmark():
    """Run quick benchmark for common scenarios."""
    suite = BenchmarkSuite(iterations=3, warmup=1)
    
    print("Quick Benchmark - madS0rt vs Python Timsort")
    print("=" * 60)
    
    test_cases = [
        (1000, 'random_strings', '1K random strings'),
        (10000, 'random_strings', '10K random strings'),
        (50000, 'random_strings', '50K random strings'),
        (10000, 'few_unique', '10K with few unique prefixes'),
        (10000, 'realistic', '10K realistic data'),
    ]
    
    all_results = []
    
    for size, dtype, desc in test_cases:
        print(f"\n{desc}:")
        data = suite.generate_data(size, dtype)
        
        py_result = suite.benchmark_python_sorted(data)
        print(f"  Python sorted(): {py_result.avg_time_ms:.2f}ms")
        
        ms_result = suite.benchmark_mads0rt(data, prefix_length=3)
        print(f"  madS0rt:         {ms_result.avg_time_ms:.2f}ms")
        
        speedup = py_result.avg_time_ms / ms_result.avg_time_ms
        print(f"  Speedup:         {speedup:.2f}x")
        
        all_results.extend([py_result, ms_result])
    
    print("\n" + "=" * 60)
    suite.print_results(all_results)
    
    return all_results


def run_full_benchmark():
    """Run full benchmark suite."""
    suite = BenchmarkSuite(iterations=5, warmup=2)
    
    print("Full Benchmark Suite")
    print("=" * 70)
    
    sizes = [100, 1000, 10000, 50000]
    all_results = []
    
    for size in sizes:
        print(f"\nSize: {size:,} items")
        data = suite.generate_data(size, 'random_strings')
        
        r1 = suite.benchmark_python_sorted(data)
        r2 = suite.benchmark_list_sort(data)
        r3 = suite.benchmark_mads0rt(data, prefix_length=2)
        r4 = suite.benchmark_mads0rt(data, prefix_length=3)
        r5 = suite.benchmark_mads0rt(data, prefix_length=4)
        r6 = suite.benchmark_mads0rt(data, adaptive=True)
        
        all_results.extend([r1, r2, r3, r4, r5, r6])
    
    suite.print_results(all_results)
    return all_results


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='madS0rt Benchmark Suite')
    parser.add_argument('--full', action='store_true', help='Run full benchmark')
    parser.add_argument('-n', '--iterations', type=int, default=3, help='Number of iterations')
    
    args = parser.parse_args()
    
    if args.full:
        run_full_benchmark()
    else:
        run_quick_benchmark()
