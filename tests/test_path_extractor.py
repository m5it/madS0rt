"""
Tests for PathExtractor and enhanced nested field access.
"""

import pytest
from dataclasses import dataclass

from madsort.extractors import (
    PathExtractor,
    FieldExtractor,
    MultiFieldExtractor,
)


@dataclass
class Address:
    street: str
    city: str
    zipcode: str


@dataclass
class Profile:
    name: str
    age: int
    address: Address


@dataclass
class User:
    id: int
    profile: Profile


class TestPathExtractor:
    """Test PathExtractor for deep path access."""
    
    def test_dict_single_level(self):
        """Access single level dict key."""
        ext = PathExtractor("name")
        data = {"name": "Alice", "age": 30}
        assert ext.extract(data) == "Alice"
    
    def test_dict_nested_two_levels(self):
        """Access nested dict: data['user']['name']."""
        ext = PathExtractor("user.name")
        data = {"user": {"name": "Bob", "age": 25}}
        assert ext.extract(data) == "Bob"
    
    def test_dict_deeply_nested(self):
        """Access deeply nested dict: data['a']['b']['c']."""
        ext = PathExtractor("a.b.c")
        data = {"a": {"b": {"c": "deep_value"}}}
        assert ext.extract(data) == "deep_value"
    
    def test_dict_with_list_path(self):
        """Access using list of keys instead of dot notation."""
        ext = PathExtractor(["user", "profile", "name"])
        data = {"user": {"profile": {"name": "Charlie"}}}
        assert ext.extract(data) == "Charlie"
    
    def test_object_attributes(self):
        """Access object attributes: obj.profile.name."""
        user = User(
            id=1,
            profile=Profile(
                name="David",
                age=35,
                address=Address("St", "City", "12345")
            )
        )
        
        ext = PathExtractor("profile.name")
        assert ext.extract(user) == "David"
    
    def test_object_deeply_nested(self):
        """Access deeply nested object attributes."""
        user = User(
            id=1,
            profile=Profile(
                name="Eve",
                age=28,
                address=Address("123 Main", "Boston", "02101")
            )
        )
        
        ext = PathExtractor("profile.address.city")
        assert ext.extract(user) == "Boston"
    
    def test_mixed_dict_and_object(self):
        """Access mix of dict and object."""
        class Container:
            def __init__(self):
                self.data = {"user": {"name": "Frank"}}
        
        container = Container()
        ext = PathExtractor("data.user.name")
        assert ext.extract(container) == "Frank"
    
    def test_missing_key_returns_default(self):
        """Return default when key not found."""
        ext = PathExtractor("missing.key", default="NOT_FOUND")
        data = {"name": "Alice"}
        assert ext.extract(data) == "NOT_FOUND"
    
    def test_missing_attribute_returns_default(self):
        """Return default when attribute not found."""
        ext = PathExtractor("profile.nonexistent", default="N/A")
        user = User(1, Profile("Name", 30, Address("St", "City", "00000")))
        assert ext.extract(user) == "N/A"
    
    def test_none_in_path_returns_default(self):
        """Return default when encountering None in path."""
        ext = PathExtractor("profile.address.city", default="NO_CITY")
        user = User(1, Profile("Name", 30, None))
        assert ext.extract(user) == "NO_CITY"
    
    def test_custom_separator(self):
        """Use custom path separator."""
        ext = PathExtractor("user/profile/name", separator="/")
        data = {"user": {"profile": {"name": "Grace"}}}
        assert ext.extract(data) == "Grace"


class TestFieldExtractorNested:
    """Test FieldExtractor with nested paths."""
    
    def test_field_extractor_with_path(self):
        """FieldExtractor now supports paths like 'user.profile.name'."""
        ext = FieldExtractor("user.profile.name")
        data = {"user": {"profile": {"name": "Henry", "age": 40}}}
        assert ext.extract(data) == "Henry"
    
    def test_field_extractor_object_path(self):
        """FieldExtractor with object attribute path."""
        user = User(
            id=1,
            profile=Profile("Ivy", 32, Address("St", "City", "00000"))
        )
        
        ext = FieldExtractor("profile.name")
        assert ext.extract(user) == "Ivy"
    
    def test_field_extractor_default_with_path(self):
        """Default value when path not found."""
        ext = FieldExtractor("user.missing.field", default="UNKNOWN")
        data = {"user": {"name": "Jack"}}
        assert ext.extract(data) == "UNKNOWN"


class TestMultiFieldExtractorNested:
    """Test MultiFieldExtractor with nested paths."""
    
    def test_multi_field_with_nested_paths(self):
        """Combine multiple nested fields."""
        data = {
            "user": {"name": "Kate", "age": 28},
            "meta": {"score": 95}
        }
        
        ext = MultiFieldExtractor(
            fields=["user.name", "meta.score"],
            separator=" - "
        )
        assert ext.extract(data) == "Kate - 95"
    
    def test_multi_field_with_objects(self):
        """MultiField with object paths."""
        user = User(
            id=1,
            profile=Profile("Leo", 45, Address("St", "Miami", "33101"))
        )
        
        ext = MultiFieldExtractor(
            fields=["profile.name", "profile.address.city"],
            separator="|"
        )
        assert ext.extract(user) == "Leo|Miami"
    
    def test_multi_field_with_missing_paths(self):
        """Handle missing paths with default."""
        data = {"user": {"name": "Mike"}}
        
        ext = MultiFieldExtractor(
            fields=["user.name", "user.missing"],
            separator=" | ",
            missing="N/A"
        )
        assert ext.extract(data) == "Mike | N/A"
    
    def test_multi_field_extract_tuple(self):
        """Extract as tuple for multi-key sorting."""
        data = {"user": {"name": "Nina", "priority": 2}}
        
        ext = MultiFieldExtractor(
            fields=["user.priority", "user.name"],
            separator=None  # Not used for tuple
        )
        result = ext.extract_tuple(data)
        assert result == (2, "Nina")


class TestRealWorldScenarios:
    """Real-world nested access scenarios."""
    
    def test_api_response_sorting(self):
        """Sort API responses by nested values."""
        responses = [
            {"data": {"user": {"profile": {"rank": 3, "name": "Charlie"}}}},
            {"data": {"user": {"profile": {"rank": 1, "name": "Alice"}}}},
            {"data": {"user": {"profile": {"rank": 2, "name": "Bob"}}}},
        ]
        
        from madsort import MadSorter
        
        sorter = MadSorter(key_func=lambda x: x["data"]["user"]["profile"]["rank"])
        result = sorter.sort(responses.copy())
        
        assert result[0]["data"]["user"]["profile"]["name"] == "Alice"
        assert result[1]["data"]["user"]["profile"]["name"] == "Bob"
        assert result[2]["data"]["user"]["profile"]["name"] == "Charlie"
    
    def test_database_record_sorting(self):
        """Sort database-like records with nested fields."""
        records = [
            {"id": 1, "info": {"department": {"name": "Engineering", "floor": 2}}},
            {"id": 2, "info": {"department": {"name": "Sales", "floor": 1}}},
            {"id": 3, "info": {"department": {"name": "Marketing", "floor": 3}}},
        ]
        
        # Extract using PathExtractor
        ext = PathExtractor("info.department.name")
        
        from madsort import MadSorter
        sorter = MadSorter(key_func=ext.extract)
        result = sorter.sort(records.copy())
        
        assert result[0]["info"]["department"]["name"] == "Engineering"
        assert result[1]["info"]["department"]["name"] == "Marketing"
        assert result[2]["info"]["department"]["name"] == "Sales"
    
    def test_log_entry_analysis(self):
        """Extract from log entries with nested metadata."""
        logs = [
            {"meta": {"service": {"name": "api", "latency": 100}}},
            {"meta": {"service": {"name": "db", "latency": 50}}},
            {"meta": {"service": {"name": "cache", "latency": 20}}},
        ]
        
        ext = PathExtractor("meta.service.latency")
        
        from madsort import MadSorter
        sorter = MadSorter(key_func=ext.extract)
        result = sorter.sort(logs.copy())
        
        latencies = [ext.extract(log) for log in result]
        assert latencies == [20, 50, 100]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
