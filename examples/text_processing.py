#!/usr/bin/env python3
"""
Text processing examples - ptext.py style usage.
Demonstrates prefix-based bucketing for word analysis.
"""

from madsort import MadSorter, FirstNCharsExtractor
from collections import Counter


def example_word_grouping():
    """Group similar words by prefix (like ptext.py)."""
    print("Word Grouping by Prefix")
    print("=" * 50)
    
    words = [
        "running", "runner", "run", "runs",
        "jumping", "jumper", "jump", "jumps",
        "walking", "walker", "walk", "walks",
        "swimming", "swimmer", "swim", "swims",
        "apple", "apricot", "application", "apply"
    ]
    
    sorter = MadSorter(prefix_length=3)
    sorter._bucket_manager.distribute(words)
    
    print(f"Total words: {len(words)}")
    print(f"Number of buckets: {len(sorter.get_buckets())}")
    print("\nGrouped by first 3 letters:")
    
    for hash_key, bucket in sorter._bucket_manager:
        if len(bucket) > 1:
            bucket_words = list(bucket)
            prefix = bucket_words[0][:3]
            print(f"  '{prefix}': {bucket_words}")
    print()


def example_prefix_coverage():
    """Calculate prefix coverage like ptext.py percentage analysis."""
    print("Prefix Coverage Analysis")
    print("=" * 50)
    
    def analyze_prefix_coverage(words, prefix_len=3):
        """Analyze how much of each word is covered by prefix."""
        results = []
        
        for word in words:
            if len(word) >= prefix_len:
                prefix = word[:prefix_len]
                coverage = (len(prefix) / len(word)) * 100
                
                results.append({
                    'word': word,
                    'prefix': prefix,
                    'coverage': coverage,
                    'word_len': len(word)
                })
        
        return results
    
    words = ["application", "apply", "apricot", "apt", "banana", "band"]
    analysis = analyze_prefix_coverage(words, prefix_len=3)
    
    print(f"{'Word':<15} {'Prefix':<8} {'Coverage':<10} {'Length':<8}")
    print("-" * 45)
    for item in analysis:
        print(f"{item['word']:<15} {item['prefix']:<8} "
              f"{item['coverage']:>6.1f}%    {item['word_len']:<8}")
    print()


def example_similarity_detection():
    """Find similar words using prefix bucketing."""
    print("Similarity Detection")
    print("=" * 50)
    
    def find_similar_words(words, prefix_len=3, min_similarity=0.5):
        """
        Find words that share common prefixes.
        Returns groups of similar words.
        """
        sorter = MadSorter(prefix_length=prefix_len)
        sorter._bucket_manager.distribute(words)
        
        similar_groups = []
        
        for hash_key, bucket in sorter._bucket_manager:
            bucket_words = list(bucket)
            if len(bucket_words) >= 2:
                # Check if words are actually similar (not just same prefix)
                prefix = bucket_words[0][:prefix_len]
                
                # Calculate average similarity
                similarities = []
                for w1 in bucket_words:
                    for w2 in bucket_words:
                        if w1 != w2:
                            # Simple similarity: common prefix length / max length
                            common = 0
                            for c1, c2 in zip(w1, w2):
                                if c1 == c2:
                                    common += 1
                                else:
                                    break
                            sim = common / max(len(w1), len(w2))
                            similarities.append(sim)
                
                avg_similarity = sum(similarities) / len(similarities) if similarities else 0
                
                if avg_similarity >= min_similarity:
                    similar_groups.append({
                        'prefix': prefix,
                        'words': bucket_words,
                        'avg_similarity': avg_similarity
                    })
        
        return similar_groups
    
    words = ["running", "runner", "runs", "run",
             "jumping", "jumper", "jumps", "jump",
             "apple", "apply", "application", "apricot",
             "cat", "car", "card", "care"]
    
    groups = find_similar_words(words, prefix_len=3, min_similarity=0.4)
    
    print(f"Found {len(groups)} similar groups:\n")
    for group in groups:
        print(f"  Prefix '{group['prefix']}' "
              f"(avg similarity: {group['avg_similarity']:.2f})")
        print(f"    Words: {group['words']}")
    print()


def example_frequency_by_prefix():
    """Count word frequencies grouped by prefix."""
    print("Frequency Analysis by Prefix")
    print("=" * 50)
    
    # Sample text data
    text_words = """
    the quick brown fox jumps over the lazy dog
    the fox runs quickly through the forest
    the dog walks slowly behind the fox
    a quick brown dog jumps over a lazy fox
    """.lower().split()
    
    # Clean words
    words = [w.strip('.,!?;:"()[]') for w in text_words if w]
    
    sorter = MadSorter(prefix_length=2)
    sorter._bucket_manager.distribute(words)
    
    print(f"Total words: {len(words)}")
    print(f"Unique words: {len(set(words))}")
    print("\nFrequency by prefix:")
    
    prefix_counts = Counter()
    for hash_key, bucket in sorter._bucket_manager:
        prefix = list(bucket)[0][:2] if bucket else "?"
        prefix_counts[prefix] += len(bucket)
    
    for prefix, count in prefix_counts.most_common():
        print(f"  '{prefix}': {count} words")
    print()


def example_adaptive_prefix():
    """Demonstrate adaptive prefix length selection."""
    print("Adaptive Prefix Length")
    print("=" * 50)
    
    from madsort import AdaptiveMadSorter
    
    # Dense data: many words, few unique prefixes
    dense_words = ['apple' + str(i) for i in range(100)]
    
    # Sparse data: few words, many unique prefixes  
    sparse_words = [''.join(__import__('random').choices(__import__('string').ascii_lowercase, k=10)) 
                    for _ in range(50)]
    
    print("Dense data (100 items, similar prefixes):")
    sorter1 = AdaptiveMadSorter(auto_adjust=True)
    sorter1.sort(dense_words)
    report1 = sorter1.get_adaptive_report()
    print(report1)
    
    print("\nSparse data (50 items, random prefixes):")
    sorter2 = AdaptiveMadSorter(auto_adjust=True)
    sorter2.sort(sparse_words)
    report2 = sorter2.get_adaptive_report()
    print(report2)


if __name__ == "__main__":
    example_word_grouping()
    example_prefix_coverage()
    example_similarity_detection()
    example_frequency_by_prefix()
    example_adaptive_prefix()
