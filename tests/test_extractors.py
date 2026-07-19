"""
Tests for madS0rt extractor system.
"""

import pytest
import re
from madsort.extractors import (
    FirstNCharsExtractor,
    LastNCharsExtractor,
    CustomRegexExtractor,
    NumericExtractor,
    MultiFieldExtractor,
    CompositeKeyExtractor,
    ConditionalExtractor,
    ChainExtractor,
    make_extractor,
    make_filename_extractor,
    make_version_extractor,
    make_date_extractor,
)


class TestFirstNCharsExtractor:
    """Test FirstNCharsExtractor."""
    
    def test_basic_extraction(self):
        ext = FirstNCharsExtractor(n=3)
        assert ext("apple") == "app"
        assert ext("banana") == "ban"
    
    def test_shorter_string(self):
        ext = FirstNCharsExtractor(n=5)
        assert ext("hi") == "hi"
    
    def test_lowercase(self):
        ext = FirstNCharsExtractor(n=3, lowercase=True)
        assert ext("APPLE") == "app"
    
    def test_preserve_case(self):
        ext = FirstNCharsExtractor(n=3, lowercase=False)
        assert ext("APPLE") == "APP"
    
    def test_padding(self):
        ext = FirstNCharsExtractor(n=3, pad='_', pad_length=5)
        assert ext("hi") == "hi___"
    
    def test_from_end(self):
        ext = FirstNCharsExtractor(n=3, from_end=True)
        assert ext("running") == "ing"
    
    def test_numeric_input(self):
        ext = FirstNCharsExtractor(n=3)
        assert ext(12345) == "123"


class TestLastNCharsExtractor:
    """Test LastNCharsExtractor."""
    
    def test_basic(self):
        ext = LastNCharsExtractor(n=3)
        assert ext("filename.txt") == "txt"
        assert ext("document.pdf") == "pdf"


class TestCustomRegexExtractor:
    """Test CustomRegexExtractor."""
    
    def test_basic_match(self):
        ext = CustomRegexExtractor(r'(\d+)')
        assert ext("abc123def") == "123"
    
    def test_named_group(self):
        ext = CustomRegexExtractor(r'(?P<num>\d+)', group='num')
        assert ext("abc123def") == "123"
    
    def test_fallback(self):
        ext = CustomRegexExtractor(r'xyz', fallback="NOTFOUND")
        assert ext("abc") == "NOTFOUND"
    
    def test_multiple_matches(self):
        ext = CustomRegexExtractor(r'\d+', multiple=True)
        result = ext("a1b2c3")
        assert result == "1|2|3"
    
    def test_with_transform(self):
        ext = CustomRegexExtractor(r'(\d+)', transform=int)
        result = ext("abc123")
        assert result == 123
        assert isinstance(result, int)


class TestNumericExtractor:
    """Test NumericExtractor."""
    
    def test_integer_extraction(self):
        ext = NumericExtractor()
        assert ext("abc123def") == 123
        assert ext("item_456") == 456
    
    def test_float_extraction(self):
        ext = NumericExtractor(as_type=float, allow_decimal=True)
        assert ext("price_12.99") == 12.99
    
    def test_negative_numbers(self):
        ext = NumericExtractor(allow_negative=True)
        assert ext("temp_-10") == -10
    
    def test_no_negative(self):
        ext = NumericExtractor(allow_negative=False)
        assert ext("temp_-10") == 10
    
    def test_default_value(self):
        ext = NumericExtractor(default=-1)
        assert ext("no numbers") == -1


class TestMultiFieldExtractor:
    """Test MultiFieldExtractor."""
    
    def test_dict_extraction(self):
        ext = MultiFieldExtractor(fields=['name', 'age'])
        data = {'name': 'John', 'age': '30'}
        assert ext(data) == "John|30"
    
    def test_object_extraction(self):
        class Person:
            name = "Jane"
            age = 25
        
        ext = MultiFieldExtractor(fields=['name', 'age'])
        assert ext(Person()) == "Jane|25"
    
    def test_missing_field(self):
        ext = MultiFieldExtractor(fields=['name', 'missing'], missing="N/A")
        data = {'name': 'John'}
        assert ext(data) == "John|N/A"
    
    def test_with_custom_extractor(self):
        ext = MultiFieldExtractor(
            fields=[
                ('name', FirstNCharsExtractor(n=4)),
                ('value', NumericExtractor())
            ]
        )
        data = {'name': 'Alexander', 'value': 'price_99'}
        assert ext(data) == "alex|99"
    
    def test_nested_attribute(self):
        class Inner:
            value = "nested"
        
        class Outer:
            inner = Inner()
        
        ext = MultiFieldExtractor(fields=['inner.value'])
        assert ext(Outer()) == "nested"


class TestCompositeKeyExtractor:
    """Test CompositeKeyExtractor."""
    
    def test_tuple_output(self):
        ext = CompositeKeyExtractor([
            FirstNCharsExtractor(n=1),
            NumericExtractor()
        ])
        result = ext("A-100")
        assert result == ('a', 100)
        assert isinstance(result, tuple)
    
    def test_multi_key_sorting(self):
        """Simulate multi-key sorting scenario."""
        data = [
            ("Alice", 30),
            ("Bob", 25),
            ("Alice", 25),
            ("Bob", 30),
        ]
        
        ext = CompositeKeyExtractor([
            lambda x: x[0],  # Name
            lambda x: x[1],  # Age
        ])
        
        sorted_data = sorted(data, key=ext)
        # Should sort by name, then by age
        assert sorted_data == [
            ("Alice", 25),
            ("Alice", 30),
            ("Bob", 25),
            ("Bob", 30),
        ]


class TestConditionalExtractor:
    """Test ConditionalExtractor."""
    
    def test_condition_matching(self):
        ext = ConditionalExtractor([
            (lambda x: x.startswith('A'), FirstNCharsExtractor(n=2)),
            (lambda x: x.startswith('B'), FirstNCharsExtractor(n=3)),
        ], default=FirstNCharsExtractor(n=1))
        
        assert ext("Apple") == "ap"
        assert ext("Banana") == "ban"
        assert ext("Cherry") == "c"  # Default


class TestChainExtractor:
    """Test ChainExtractor."""
    
    def test_chaining(self):
        # First extract number, then convert to string and take first char
        ext = ChainExtractor([
            NumericExtractor(),
            lambda x: str(x),
            FirstNCharsExtractor(n=1)
        ])
        
        assert ext("price_123") == "1"


class TestFactoryFunctions:
    """Test factory functions."""
    
    def test_make_extractor(self):
        ext = make_extractor('prefix', n=4)
        assert ext("hello") == "hell"
        
        ext = make_extractor('numeric', default=-1)
        assert ext("abc") == -1
    
    def test_make_filename_extractor(self):
        ext = make_filename_extractor()
        assert ext({'name': 'document', 'ext': 'pdf'}) == "document.pdf"
    
    def test_make_version_extractor(self):
        ext = make_version_extractor()
        # Extracts numbers from version string
        result = ext("v1.2.3")
        assert isinstance(result, tuple)
    
    def test_make_date_extractor(self):
        ext = make_date_extractor()
        result = ext("2024-03-15")
        assert result == (2024, 3, 15)


class TestIntegration:
    """Integration tests with BucketManager."""
    
    def test_with_bucket_manager(self):
        from madsort.bucket import BucketManager
        
        # Create manager with custom extractor
        extractor = CompositeKeyExtractor([
            FirstNCharsExtractor(n=2),
            NumericExtractor(default=0)
        ])
        
        # Items with same prefix but different numbers
        items = [
            "AB-100",
            "AB-50",
            "CD-200",
            "AB-75",
        ]
        
        manager = BucketManager(
            prefix_length=2,
            key_func=lambda x: extractor(x)[0]  # Use first part of composite
        )
        
        manager.distribute(items)
        
        # AB items should be in same bucket
        # CD item should be in different bucket
        assert manager.get_stats()['num_buckets'] == 2


class TestRealWorldScenarios:
    """Real-world usage scenarios."""
    
    def test_log_file_parsing(self):
        """Extract timestamp and level from log lines."""
        ext = MultiFieldExtractor(
            fields=[
                ('timestamp', CustomRegexExtractor(r'\[(.*?)\]', group=1)),
                ('level', CustomRegexExtractor(r'\[.*?\]\s*(\w+)', group=1)),
            ],
            separator=' '
        )
        
        log_line = "[2024-03-15 10:30:00] ERROR Something went wrong"
        result = ext(log_line)
        assert "2024-03-15 10:30:00" in result
        assert "ERROR" in result
    
    def test_product_code_sorting(self):
        """Extract category and serial number from product codes."""
        ext = CompositeKeyExtractor([
            FirstNCharsExtractor(n=2),  # Category
            NumericExtractor(),  # Serial number
        ])
        
        products = ["EL-1001", "EL-502", "ME-100", "EL-999"]
        sorted_products = sorted(products, key=ext)
        
        # Should be sorted by category, then by serial
        assert sorted_products[0] == "EL-502"
        assert sorted_products[1] == "EL-999"
        assert sorted_products[2] == "EL-1001"
    
    def test_email_sorting(self):
        """Extract domain from email for grouping."""
        ext = CustomRegexExtractor(r'@([\w.]+)', group=1)
        
        emails = [
            "alice@gmail.com",
            "bob@yahoo.com",
            "charlie@gmail.com",
        ]
        
        domains = [ext(e) for e in emails]
        assert domains == ["gmail.com", "yahoo.com", "gmail.com"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
