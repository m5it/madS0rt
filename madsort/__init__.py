"""
madS0rt - High-performance hybrid sorting library
using prefix-based bucketing with pluggable key extractors.
"""

__version__ = "0.1.0"
__author__ = "w4d4f4k"

from .bucket import Bucket, BucketManager
from .hash_utils import crc32_hash, xxhash_hash, HashProvider, get_hash_provider
from .extractors import (
    BaseExtractor,
    FirstNCharsExtractor,
    LastNCharsExtractor,
    CustomRegexExtractor,
    NumericExtractor,
    PathExtractor,
    MultiFieldExtractor,
    CompositeKeyExtractor,
    ConditionalExtractor,
    ChainExtractor,
    PrefixExtractor,
    RegexExtractor,
    make_extractor,
    make_filename_extractor,
    make_version_extractor,
    make_date_extractor,
)
from .sorter import MadSorter, SortStrategy, madsort, madsorted
from .adaptive import (
    DistributionAnalyzer,
    LoadBalancer,
    AdaptiveBucketManager,
    AdaptiveMadSorter,
)
from .gpu_backend import (
    gpu_available,
    GPUBackend,
    gpu_sort,
)

__all__ = [
    # Core
    'Bucket',
    'BucketManager',
    # Hash
    'crc32_hash',
    'xxhash_hash',
    'HashProvider',
    'get_hash_provider',
    # Extractors
    'BaseExtractor',
    'FirstNCharsExtractor',
    'LastNCharsExtractor',
    'CustomRegexExtractor',
    'NumericExtractor',
    'PathExtractor',
    'MultiFieldExtractor',
    'CompositeKeyExtractor',
    'ConditionalExtractor',
    'ChainExtractor',
    'PrefixExtractor',
    'RegexExtractor',
    # Factories
    'make_extractor',
    'make_filename_extractor',
    'make_version_extractor',
    'make_date_extractor',
    # Sorter
    'MadSorter',
    'SortStrategy',
    'madsort',
    'madsorted',
    # Adaptive
    'DistributionAnalyzer',
    'LoadBalancer',
    'AdaptiveBucketManager',
    'AdaptiveMadSorter',
    # GPU
    'gpu_available',
    'GPUBackend',
    'gpu_sort',
]
