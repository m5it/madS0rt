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
    LoadBalancer,
    AdaptiveBucketManager,
    AdaptiveMadSorter,
)

__all__ = [
    'Bucket',
    'BucketManager',
    'crc32_hash',
    'xxhash_hash',
    'HashProvider',
    'get_hash_provider',
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
    'make_extractor',
    'make_filename_extractor',
    'make_version_extractor',
    'make_date_extractor',
    'MadSorter',
    'SortStrategy',
    'madsort',
    'madsorted',
    'DistributionAnalyzer',
    'LoadBalancer',
    'AdaptiveBucketManager',
    'AdaptiveMadSorter',
]
    'MadSorter',
    'SortStrategy',
    'madsort',
    'madsorted',
    'DistributionAnalyzer',
    'LoadBalancer',
    'AdaptiveBucketManager',
    'AdaptiveMadSorter',
]
