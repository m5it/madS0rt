"""
Tests for madS0rt bucket implementation.
"""

import pytest
from madsort.bucket import Bucket, BucketManager
from madsort.hash_utils import crc32_hash, get_hash_provider


class TestBucket:
    """Test Bucket class."""
    
    def test_basic_creation(self):
        bucket = Bucket(1, 0x12345678)
        assert bucket.bucket_id == 1
        assert bucket.hash_key == 0x12345678
        assert len(bucket) == 0
    
    def test_add_and_sort(self):
        bucket = Bucket(1, 0x12345678)
        bucket.add("zebra")
        bucket.add("apple")
        bucket.add("banana")
        
        assert len(bucket) == 3
        assert not bucket._sorted
        
        bucket.sort()
        assert bucket.items == ["apple", "banana", "zebra"]
        assert bucket._sorted
    
    def test_sort_with_key(self):
        bucket = Bucket(1, 0x12345678, key_func=lambda x: x[1])
        bucket.add((1, "zebra"))
        bucket.add((2, "apple"))
        bucket.add((3, "banana"))
        
        bucket.sort()
        assert bucket.items == [(2, "apple"), (3, "banana"), (1, "zebra")]
    
    def test_merge_buckets(self):
        b1 = Bucket(1, 0x12345678)
        b1.add_batch([1, 3, 5])
        b1.sort()
        
        b2 = Bucket(2, 0x12345679)
        b2.add_batch([2, 4, 6])
        b2.sort()
        
        b1.merge(b2, preserve_order=True)
        assert b1.items == [1, 2, 3, 4, 5, 6]
    
    def test_split_bucket(self):
        bucket = Bucket(1, 0x12345678)
        bucket.add_batch([1, 2, 3, 4, 5, 6])
        
        even, odd = bucket.split(lambda x: x % 2 == 0)
        assert list(even.items) == [2, 4, 6]
        assert list(odd.items) == [1, 3, 5]


class TestBucketManager:
    """Test BucketManager class."""
    
    def test_basic_distribution(self):
        manager = BucketManager(prefix_length=2)
        words = ["apple", "apricot", "banana", "blueberry", "cherry"]
        
        buckets = manager.distribute(words)
        
        # Should create buckets based on first 2 letters
        assert len(buckets) > 0
        assert manager.get_stats()['total_items'] == 5
    
    def test_prefix_hashing(self):
        manager = BucketManager(prefix_length=3)
        
        # These should go to same bucket (same first 3 letters)
        manager.add("apple")
        manager.add("apricot")
        manager.add("application")
        
        # These should go to different bucket
        manager.add("banana")
        
        stats = manager.get_stats()
        assert stats['num_buckets'] == 2
        assert stats['total_items'] == 4
    
    def test_crc32_compatibility(self):
        """Ensure our CRC32 matches ptext.py expectations."""
        manager = BucketManager(
            hash_provider=get_hash_provider("crc32"),
            prefix_length=3
        )
        
        # Test specific hash values
        prefix = "app"
        expected_hash = crc32_hash(prefix)
        
        bucket_id = manager.add("apple")
        assert bucket_id == expected_hash
    
    def test_get_all_items(self):
        manager = BucketManager(prefix_length=1)
        words = ["zebra", "apple", "banana", "apricot"]
        
        manager.distribute(words)
        items = manager.get_all_items(sorted_buckets=True)
        
        # Should be sorted within buckets, buckets in hash order
        assert len(items) == 4
        # 'a' words should be together and sorted
        assert items[0] in ["apple", "apricot"]
        assert items[1] in ["apple", "apricot"]
    
    def test_merge_all(self):
        manager = BucketManager(prefix_length=1)
        words = ["zebra", "apple", "banana"]
        
        manager.distribute(words)
        for _, bucket in manager:
            bucket.sort()
        
        merged = manager.merge_all(preserve_order=True)
        assert len(merged) == 3
    
    def test_bucket_splitting(self):
        manager = BucketManager(
            prefix_length=1,
            max_bucket_size=2
        )
        
        # Add 5 items with same first letter
        words = ["apple", "apricot", "application", "apply", "apt"]
        for w in words:
            manager.add(w)
        
        # Should have split the bucket
        stats = manager.get_stats()
        assert stats['num_buckets'] >= 2
    
    def test_clear(self):
        manager = BucketManager()
        manager.add("test")
        
        assert manager.get_stats()['total_items'] == 1
        manager.clear()
        assert manager.get_stats()['total_items'] == 0
        assert len(manager) == 0


class TestIntegration:
    """Integration tests mimicking ptext.py usage."""
    
    def test_word_frequency_style(self):
        """
        Simulate ptext.py word counting and grouping.
        Words with same prefix should bucket together for fast comparison.
        """
        words = [
            "running", "runner", "run", "runs",
            "jumping", "jumper", "jump", "jumps",
            "walking", "walker", "walk", "walks"
        ]
        
        manager = BucketManager(prefix_length=3)
        manager.distribute(words)
        
        # Group by first 3 letters for similarity detection
        for hash_key, bucket in manager:
            if len(bucket) > 1:
                # These words share prefix and can be compared for similarity
                prefix_words = list(bucket)
                # All should start with same 3 letters
                prefixes = {w[:3] for w in prefix_words}
                assert len(prefixes) == 1
    
    def test_large_dataset_performance(self):
        """Test with larger dataset to ensure scalability."""
        import random
        import string
        
        # Generate random words
        words = []
        for _ in range(1000):
            length = random.randint(3, 10)
            word = ''.join(random.choices(string.ascii_lowercase, k=length))
            words.append(word)
        
        manager = BucketManager(prefix_length=3)
        buckets = manager.distribute(words)
        
        stats = manager.get_stats()
        assert stats['total_items'] == 1000
        
        # Verify all items retrievable
        retrieved = manager.get_all_items()
        assert len(retrieved) == 1000
        assert set(retrieved) == set(words)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
