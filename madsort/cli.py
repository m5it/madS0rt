"""
Command-line interface for madS0rt.
Provides file sorting with various options and performance reporting.
"""

import argparse
import sys
import time
import os
from typing import Optional, List

from .sorter import MadSorter, madsort
from .adaptive import AdaptiveMadSorter
from .extractors import make_extractor, NumericExtractor


def read_file(filepath: str, encoding: str = 'utf-8') -> List[str]:
    """Read lines from file."""
    with open(filepath, 'r', encoding=encoding, errors='ignore') as f:
        return [line.rstrip('\n\r') for line in f]


def write_file(filepath: str, lines: List[str], encoding: str = 'utf-8') -> None:
    """Write lines to file."""
    with open(filepath, 'w', encoding=encoding) as f:
        for line in lines:
            f.write(line + '\n')


def sort_file(
    input_file: str,
    output_file: Optional[str] = None,
    key: Optional[str] = None,
    reverse: bool = False,
    adaptive: bool = False,
    prefix_length: int = 3,
    numeric: bool = False,
    verbose: bool = False
) -> dict:
    """Sort a file using madS0rt."""
    if verbose:
        print(f"Reading {input_file}...")
    
    lines = read_file(input_file)
    
    if verbose:
        print(f"Sorting {len(lines):,} lines...")
    
    key_func = None
    if numeric:
        key_func = NumericExtractor(default=0)
    elif key:
        try:
            key_func = make_extractor('regex', pattern=key, group=0)
        except:
            pass
    
    if adaptive:
        sorter = AdaptiveMadSorter(
            initial_prefix_length=prefix_length,
            key_func=key_func,
            auto_adjust=True,
            enable_load_balance=True
        )
        result = sorter.sort(lines, reverse=reverse)
        stats = sorter.get_stats()
    else:
        sorter = MadSorter(
            prefix_length=prefix_length,
            key_func=key_func,
            copy_mode=True
        )
        result = sorter.sort(lines, reverse=reverse)
        stats = sorter.get_stats()
    
    output_path = output_file or input_file
    if verbose:
        print(f"Writing to {output_path}...")
    
    write_file(output_path, result)
    
    if verbose:
        print(f"Done! Sorted {len(lines):,} lines.")
        if 'total_time_ms' in stats:
            print(f"Time: {stats['total_time_ms']:.2f}ms")
    
    return stats


def create_parser() -> argparse.ArgumentParser:
    """Create argument parser."""
    parser = argparse.ArgumentParser(
        prog='mads0rt',
        description='High-performance hybrid sorting with prefix-based bucketing',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  mads0rt file.txt                    Sort file in-place
  mads0rt input.txt -o output.txt     Sort to new file
  mads0rt file.txt -r                 Reverse sort
  mads0rt file.txt -n                 Numeric sort
  mads0rt file.txt -a                 Use adaptive mode
  mads0rt file.txt -p 2               Use 2-char prefix bucketing
        """
    )
    
    parser.add_argument('input', help='Input file to sort')
    parser.add_argument('-o', '--output', help='Output file (default: overwrite input)')
    parser.add_argument('-r', '--reverse', action='store_true', help='Reverse sort order')
    parser.add_argument('-n', '--numeric', action='store_true', help='Numeric sort')
    parser.add_argument('-a', '--adaptive', action='store_true', help='Use adaptive bucketing')
    parser.add_argument('-p', '--prefix-length', type=int, default=3, 
                        help='Prefix length for bucketing (default: 3)')
    parser.add_argument('-k', '--key', help='Key extraction regex pattern')
    parser.add_argument('-v', '--verbose', action='store_true', help='Verbose output')
    parser.add_argument('--version', action='version', version='%(prog)s 0.1.0')
    
    return parser


def main(args: Optional[List[str]] = None) -> int:
    """Main entry point."""
    parser = create_parser()
    parsed = parser.parse_args(args)
    
    if not os.path.exists(parsed.input):
        print(f"Error: File not found: {parsed.input}", file=sys.stderr)
        return 1
    
    try:
        stats = sort_file(
            input_file=parsed.input,
            output_file=parsed.output,
            key=parsed.key,
            reverse=parsed.reverse,
            adaptive=parsed.adaptive,
            prefix_length=parsed.prefix_length,
            numeric=parsed.numeric,
            verbose=parsed.verbose
        )
        
        if parsed.verbose:
            print(f"\nStatistics:")
            print(f"  Items: {stats.get('total_items', 'N/A')}")
            print(f"  Buckets: {stats.get('num_buckets', 'N/A')}")
            print(f"  Sort time: {stats.get('sort_time_ms', 0):.2f}ms")
            print(f"  Merge time: {stats.get('merge_time_ms', 0):.2f}ms")
            print(f"  Total time: {stats.get('total_time_ms', 0):.2f}ms")
        
        return 0
        
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


if __name__ == '__main__':
    sys.exit(main())
