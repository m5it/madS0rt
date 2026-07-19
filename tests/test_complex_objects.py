"""
Tests for madS0rt sorting of complex nested objects.
Verifies that MadSorter can handle objects, nested dicts, and mixed types.
"""

import pytest
from dataclasses import dataclass
from typing import Optional

from madsort import MadSorter, madsorted
from madsort.extractors import FieldExtractor, MultiFieldExtractor, CompositeKeyExtractor


# Test Data Classes
@dataclass
class Address:
    street: str
    city: str
    zipcode: str


@dataclass
class Person:
    name: str
    age: int
    address: Address
    
    def __repr__(self):
        return f"Person({self.name}, {self.age})"


@dataclass
class Company:
    name: str
    ceo: Optional[Person]
    revenue: float


# Test class with custom comparison
class ComparableItem:
    def __init__(self, value: int, label: str):
        self.value = value
        self.label = label
    
    def __lt__(self, other):
        return self.value < other.value
    
    def __eq__(self, other):
        return self.value == other.value and self.label == other.label
    
    def __repr__(self):
        return f"Item({self.value}, {self.label})"


class TestNestedObjects:
    """Test sorting objects with nested attributes."""
    
    def test_sort_by_nested_attribute(self):
        """Sort objects by nested attribute (person.address.city)."""
        people = [
            Person("Alice", 30, Address("123 Main", "Boston", "02101")),
            Person("Bob", 25, Address("456 Oak", "Atlanta", "30301")),
            Person("Charlie", 35, Address("789 Pine", "Chicago", "60601")),
        ]
        
        sorter = MadSorter(key_func=lambda p: p.address.city)
        result = sorter.sort(people.copy())
        
        # Should be sorted by city: Atlanta, Boston, Chicago
        assert result[0].address.city == "Atlanta"
        assert result[1].address.city == "Boston"
        assert result[2].address.city == "Chicago"
    
    def test_sort_by_deeply_nested_attribute(self):
        """Sort by deeply nested attribute (company.ceo.address.city)."""
        companies = [
            Company("TechCorp", Person("John", 50, Address("1 Tech", "Seattle", "98101")), 1000000.0),
            Company("FinanceInc", Person("Jane", 45, Address("2 Finance", "Miami", "33101")), 2000000.0),
            Company("RetailCo", Person("Bob", 40, Address("3 Retail", "Denver", "80201")), 500000.0),
        ]
        
        sorter = MadSorter(key_func=lambda c: c.ceo.address.city)
        result = sorter.sort(companies.copy())
        
        cities = [c.ceo.address.city for c in result]
        assert cities == ["Denver", "Miami", "Seattle"]
    
    def test_sort_objects_with_none_values(self):
        """Handle objects with None nested attributes."""
        companies = [
            Company("A", Person("Alice", 30, Address("St", "Boston", "02101")), 100.0),
            Company("B", None, 200.0),  # No CEO
            Company("C", Person("Bob", 25, Address("St", "Atlanta", "30301")), 150.0),
        ]
        
        # Sort by CEO name, handling None
        def get_ceo_name(c):
            return c.ceo.name if c.ceo else ""
        
        sorter = MadSorter(key_func=get_ceo_name)
        result = sorter.sort(companies.copy())
        
        # Company B (no CEO) should be first (empty string), then Alice, then Bob
        assert result[0].name == "B"
        assert result[1].ceo.name == "Alice"
        assert result[2].ceo.name == "Bob"


class TestNestedDictionaries:
    """Test sorting dictionaries with nested values."""
    
    def test_sort_by_nested_dict_value(self):
        """Sort list of dicts by nested value (data['user']['name'])."""
        data = [
            {"id": 1, "user": {"name": "Charlie", "age": 30}},
            {"id": 2, "user": {"name": "Alice", "age": 25}},
            {"id": 3, "user": {"name": "Bob", "age": 35}},
        ]
        
        sorter = MadSorter(key_func=lambda x: x["user"]["name"])
        result = sorter.sort(data.copy())
        
        assert result[0]["user"]["name"] == "Alice"
        assert result[1]["user"]["name"] == "Bob"
        assert result[2]["user"]["name"] == "Charlie"
    
    def test_sort_by_deeply_nested_dict(self):
        """Sort by deeply nested dict value."""
        data = [
            {"company": {"department": {"manager": {"name": "Zack"}}}},
            {"company": {"department": {"manager": {"name": "Anna"}}}},
            {"company": {"department": {"manager": {"name": "Mike"}}}},
        ]
        
        sorter = MadSorter(key_func=lambda x: x["company"]["department"]["manager"]["name"])
        result = sorter.sort(data.copy())
        
        names = [d["company"]["department"]["manager"]["name"] for d in result]
        assert names == ["Anna", "Mike", "Zack"]
    
    def test_sort_mixed_dict_depths(self):
        """Handle dicts with varying nesting depths."""
        data = [
            {"name": "Direct", "value": 3},
            {"nested": {"name": "Deep", "value": 1}},
            {"name": "Shallow", "value": 2},
        ]
        
        # Sort by 'value' with fallback
        def get_value(d):
            if "value" in d:
                return d["value"]
            elif "nested" in d:
                return d["nested"]["value"]
            return 0
        
        sorter = MadSorter(key_func=get_value)
        result = sorter.sort(data.copy())
        
        assert result[0].get("value", result[0].get("nested", {}).get("value")) == 1
        assert result[1].get("value", result[1].get("nested", {}).get("value")) == 2
        assert result[2].get("value", result[2].get("nested", {}).get("value")) == 3


class TestMixedTypes:
    """Test sorting mixed types with custom keys."""
    
    def test_sort_mixed_types_with_key(self):
        """Sort mixed types by converting to common format."""
        data = [
            {"name": "Alice", "priority": 2},
            ("Bob", {"priority": 1}),  # Tuple with dict
            ["Charlie", {"priority": 3}],  # List with dict
        ]
        
        # Extract priority from various structures
        def get_priority(item):
            if isinstance(item, dict):
                return item["priority"]
            elif isinstance(item, (tuple, list)):
                return item[1]["priority"]
            return 0
        
        sorter = MadSorter(key_func=get_priority)
        result = sorter.sort(data.copy())
        
        priorities = [get_priority(r) for r in result]
        assert priorities == [1, 2, 3]
    
    def test_sort_objects_and_dicts_together(self):
        """Sort mix of objects and dictionaries."""
        person_obj = Person("David", 40, Address("St", "City", "00000"))
        person_dict = {"name": "Eve", "age": 30, "type": "dict"}
        
        data = [
            {"name": "Charlie", "age": 35},
            person_obj,
            {"name": "Alice", "age": 25},
            person_dict,
        ]
        
        # Sort by name
        def get_name(item):
            if isinstance(item, Person):
                return item.name
            return item.get("name", "")
        
        sorter = MadSorter(key_func=get_name)
        result = sorter.sort(data.copy())
        
        names = [get_name(r) for r in result]
        assert names == ["Alice", "Charlie", "David", "Eve"]


class TestComparableClasses:
    """Test sorting classes with custom comparison methods."""
    
    def test_sort_with_lt_method(self):
        """Sort objects using __lt__ method."""
        items = [
            ComparableItem(10, "B"),
            ComparableItem(5, "A"),
            ComparableItem(15, "C"),
        ]
        
        # Python's sort should use __lt__
        sorter = MadSorter()
        result = sorter.sort(items.copy())
        
        values = [r.value for r in result]
        assert values == [5, 10, 15]
    
    def test_sort_with_nested_comparable(self):
        """Sort objects containing comparable objects."""
        @dataclass
        class Container:
            item: ComparableItem
            label: str
        
        containers = [
            Container(ComparableItem(30, "Z"), "Third"),
            Container(ComparableItem(10, "X"), "First"),
            Container(ComparableItem(20, "Y"), "Second"),
        ]
        
        sorter = MadSorter(key_func=lambda c: c.item)
        result = sorter.sort(containers.copy())
        
        assert result[0].label == "First"
        assert result[1].label == "Second"
        assert result[2].label == "Third"


class TestCompositeAndMultiField:
    """Test composite key extraction with nested data."""
    
    def test_composite_with_nested_fields(self):
        """Use CompositeKeyExtractor with nested data."""
        data = [
            {"category": "B", "priority": 2, "name": "Item2"},
            {"category": "A", "priority": 1, "name": "Item1"},
            {"category": "A", "priority": 2, "name": "Item3"},
        ]
        
        # Sort by category, then priority
        composite = CompositeKeyExtractor([
            lambda x: x["category"],
            lambda x: x["priority"],
        ])
        
        sorter = MadSorter(key_func=composite)
        result = sorter.sort(data.copy())
        
        # A1, A2, B2
        assert result[0]["name"] == "Item1"
        assert result[1]["name"] == "Item3"
        assert result[2]["name"] == "Item2"
    
    def test_multifield_with_objects(self):
        """Use MultiFieldExtractor with objects."""
        extractor = MultiFieldExtractor(
            fields=[
                ('address.city', FieldExtractor('city', default='')),
                ('name', FieldExtractor('name', default='')),
            ],
            separator='-'
        )
        
        people = [
            Person("Bob", 25, Address("St", "Atlanta", "30301")),
            Person("Alice", 30, Address("St", "Boston", "02101")),
        ]
        
        # Extract combined key
        keys = [extractor(p) for p in people]
        assert "Atlanta" in keys[0]
        assert "Boston" in keys[1]


class TestRealWorldScenarios:
    """Real-world complex sorting scenarios."""
    
    def test_sort_log_entries(self):
        """Sort log entries with nested metadata."""
        logs = [
            {"timestamp": "2024-03-15", "level": "ERROR", "meta": {"service": "api", "latency": 500}},
            {"timestamp": "2024-03-15", "level": "INFO", "meta": {"service": "db", "latency": 100}},
            {"timestamp": "2024-03-15", "level": "ERROR", "meta": {"service": "db", "latency": 200}},
        ]
        
        # Sort by level, then by service
        sorter = MadSorter(key_func=lambda x: (x["level"], x["meta"]["service"]))
        result = sorter.sort(logs.copy())
        
        # ERROR-api, ERROR-db, INFO-db
        assert result[0]["meta"]["service"] == "api"
        assert result[1]["meta"]["service"] == "db"
        assert result[2]["level"] == "INFO"
    
    def test_sort_ecommerce_products(self):
        """Sort products with nested pricing info."""
        products = [
            {"name": "Laptop", "pricing": {"base": 1000, "discount": 0.1}},
            {"name": "Phone", "pricing": {"base": 500, "discount": 0.2}},
            {"name": "Tablet", "pricing": {"base": 800, "discount": 0.0}},
        ]
        
        # Sort by final price (base * (1 - discount))
        def get_final_price(p):
            base = p["pricing"]["base"]
            discount = p["pricing"]["discount"]
            return base * (1 - discount)
        
        sorter = MadSorter(key_func=get_final_price)
        result = sorter.sort(products.copy())
        
        # Phone: 400, Tablet: 800, Laptop: 900
        assert result[0]["name"] == "Phone"
        assert result[1]["name"] == "Tablet"
        assert result[2]["name"] == "Laptop"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
