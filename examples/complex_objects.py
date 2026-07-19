#!/usr/bin/env python3
"""
Complex object sorting examples for madS0rt.
Demonstrates sorting nested objects, dictionaries, and mixed data structures.
"""

from dataclasses import dataclass
from typing import Optional, List, Dict, Any
import json

from madsort import MadSorter, madsorted, AdaptiveMadSorter
from madsort.extractors import (
    FieldExtractor,
    MultiFieldExtractor,
    CompositeKeyExtractor,
    NumericExtractor,
)


# =============================================================================
# Example Data Structures
# =============================================================================

@dataclass
class Address:
    street: str
    city: str
    country: str
    zipcode: str


@dataclass
class Department:
    name: str
    code: str
    budget: float


@dataclass
class Employee:
    id: int
    name: str
    salary: float
    department: Department
    address: Address
    manager: Optional['Employee'] = None
    
    def __repr__(self):
        return f"Employee({self.name}, {self.department.name})"


# =============================================================================
# Example 1: Database-like Records with Nested Fields
# =============================================================================

def example_database_records():
    """Sort employee records by nested department and location fields."""
    print("=" * 70)
    print("Example 1: Database-like Records with Nested Fields")
    print("=" * 70)
    
    employees = [
        Employee(
            id=1, name="Alice Johnson", salary=75000.0,
            department=Department("Engineering", "ENG", 1000000.0),
            address=Address("123 Tech St", "San Francisco", "USA", "94105")
        ),
        Employee(
            id=2, name="Bob Smith", salary=65000.0,
            department=Department("Sales", "SAL", 500000.0),
            address=Address("456 Market St", "New York", "USA", "10001")
        ),
        Employee(
            id=3, name="Carol White", salary=80000.0,
            department=Department("Engineering", "ENG", 1000000.0),
            address=Address("789 Code Ave", "Seattle", "USA", "98101")
        ),
        Employee(
            id=4, name="David Brown", salary=70000.0,
            department=Department("Marketing", "MKT", 300000.0),
            address=Address("321 Ad Blvd", "Boston", "USA", "02101")
        ),
    ]
    
    print(f"\nEmployees ({len(employees)} total):")
    for emp in employees:
        print(f"  {emp.name} - {emp.department.name} - {emp.address.city}")
    
    # Sort by department name, then by employee name
    print("\n--- Sorted by Department, then Name ---")
    sorter = MadSorter(key_func=lambda e: (e.department.name, e.name))
    sorted_emps = sorter.sort(employees.copy())
    
    for emp in sorted_emps:
        print(f"  {emp.department.name:15} | {emp.name:15} | {emp.address.city}")
    
    # Sort by city (nested in address)
    print("\n--- Sorted by City ---")
    sorter = MadSorter(key_func=lambda e: e.address.city)
    sorted_by_city = sorter.sort(employees.copy())
    
    for emp in sorted_by_city:
        print(f"  {emp.address.city:15} | {emp.name}")
    
    # Sort by department budget (descending), then salary
    print("\n--- Sorted by Dept Budget (desc), then Salary ---")
    sorter = MadSorter(key_func=lambda e: (-e.department.budget, -e.salary))
    sorted_by_budget = sorter.sort(employees.copy())
    
    for emp in sorted_by_budget:
        print(f"  Budget: ${emp.department.budget:>10,.0f} | "
              f"Salary: ${emp.salary:>8,.0f} | {emp.name}")


# =============================================================================
# Example 2: JSON-like Nested Dictionaries
# =============================================================================

def example_json_nested():
    """Sort JSON-like nested dictionary structures."""
    print("\n" + "=" * 70)
    print("Example 2: JSON-like Nested Dictionaries")
    print("=" * 70)
    
    # Simulating API response data
    api_responses = [
        {
            "user": {
                "id": 102,
                "profile": {
                    "name": "Charlie",
                    "settings": {
                        "theme": "dark",
                        "notifications": True
                    }
                },
                "stats": {
                    "login_count": 45,
                    "last_active": "2024-03-10"
                }
            },
            "timestamp": "2024-03-15T10:30:00Z"
        },
        {
            "user": {
                "id": 101,
                "profile": {
                    "name": "Alice",
                    "settings": {
                        "theme": "light",
                        "notifications": False
                    }
                },
                "stats": {
                    "login_count": 120,
                    "last_active": "2024-03-14"
                }
            },
            "timestamp": "2024-03-15T09:00:00Z"
        },
        {
            "user": {
                "id": 103,
                "profile": {
                    "name": "Bob",
                    "settings": {
                        "theme": "dark",
                        "notifications": True
                    }
                },
                "stats": {
                    "login_count": 78,
                    "last_active": "2024-03-13"
                }
            },
            "timestamp": "2024-03-15T11:15:00Z"
        },
    ]
    
    print(f"\nAPI Responses ({len(api_responses)} items):")
    for resp in api_responses:
        name = resp["user"]["profile"]["name"]
        logins = resp["user"]["stats"]["login_count"]
        print(f"  {name}: {logins} logins")
    
    # Sort by nested user name
    print("\n--- Sorted by User Name ---")
    sorter = MadSorter(key_func=lambda x: x["user"]["profile"]["name"])
    sorted_by_name = sorter.sort(api_responses.copy())
    
    for resp in sorted_by_name:
        name = resp["user"]["profile"]["name"]
        print(f"  {name}")
    
    # Sort by login count (descending)
    print("\n--- Sorted by Login Count (desc) ---")
    sorter = MadSorter(key_func=lambda x: -x["user"]["stats"]["login_count"])
    sorted_by_logins = sorter.sort(api_responses.copy())
    
    for resp in sorted_by_logins:
        name = resp["user"]["profile"]["name"]
        logins = resp["user"]["stats"]["login_count"]
        print(f"  {logins:3d} logins - {name}")
    
    # Sort by theme, then by name
    print("\n--- Sorted by Theme, then Name ---")
    sorter = MadSorter(key_func=lambda x: (
        x["user"]["profile"]["settings"]["theme"],
        x["user"]["profile"]["name"]
    ))
    sorted_by_theme = sorter.sort(api_responses.copy())
    
    for resp in sorted_by_theme:
        theme = resp["user"]["profile"]["settings"]["theme"]
        name = resp["user"]["profile"]["name"]
        print(f"  Theme: {theme:5} | {name}")


# =============================================================================
# Example 3: Parent-Child Relationships
# =============================================================================

def example_parent_child():
    """Sort objects with parent-child relationships (org chart)."""
    print("\n" + "=" * 70)
    print("Example 3: Parent-Child Relationships (Organization Chart)")
    print("=" * 70)
    
    # Create org chart
    ceo = Employee(
        id=1, name="John CEO", salary=500000.0,
        department=Department("Executive", "EXE", 0),
        address=Address("1 Corp Way", "New York", "USA", "10001")
    )
    
    vp_eng = Employee(
        id=2, name="Jane VP Engineering", salary=250000.0,
        department=Department("Engineering", "ENG", 5000000),
        address=Address("2 Tech Blvd", "San Francisco", "USA", "94105"),
        manager=ceo
    )
    
    vp_sales = Employee(
        id=3, name="Bob VP Sales", salary=220000.0,
        department=Department("Sales", "SAL", 3000000),
        address=Address("3 Sales St", "Chicago", "USA", "60601"),
        manager=ceo
    )
    
    dev1 = Employee(
        id=4, name="Alice Developer", salary=120000.0,
        department=Department("Engineering", "ENG", 5000000),
        address=Address("4 Code Lane", "Seattle", "USA", "98101"),
        manager=vp_eng
    )
    
    dev2 = Employee(
        id=5, name="Charlie Developer", salary=110000.0,
        department=Department("Engineering", "ENG", 5000000),
        address=Address("5 Dev Ave", "Austin", "USA", "78701"),
        manager=vp_eng
    )
    
    sales1 = Employee(
        id=6, name="Diana Sales Rep", salary=90000.0,
        department=Department("Sales", "SAL", 3000000),
        address=Address("6 Deal St", "Miami", "USA", "33101"),
        manager=vp_sales
    )
    
    org_chart = [dev1, ceo, sales1, vp_eng, dev2, vp_sales]
    
    print(f"\nOrg Chart ({len(org_chart)} employees):")
    for emp in org_chart:
        manager = emp.manager.name if emp.manager else "None"
        print(f"  {emp.name:20} | Manager: {manager}")
    
    # Sort by manager's name (group by team)
    print("\n--- Sorted by Manager (grouped by team) ---")
    
    def get_manager_name(emp):
        return emp.manager.name if emp.manager else "AAA_CEO"  # CEO first
    
    sorter = MadSorter(key_func=get_manager_name)
    sorted_by_manager = sorter.sort(org_chart.copy())
    
    for emp in sorted_by_manager:
        manager = emp.manager.name if emp.manager else "CEO"
        print(f"  Manager: {manager:15} | {emp.name}")
    
    # Sort by department, then by level (distance from CEO)
    print("\n--- Sorted by Department, then Level ---")
    
    def get_level(emp):
        level = 0
        current = emp
        while current.manager:
            level += 1
            current = current.manager
        return level
    
    sorter = MadSorter(key_func=lambda e: (e.department.name, get_level(e)))
    sorted_by_dept = sorter.sort(org_chart.copy())
    
    for emp in sorted_by_dept:
        level = get_level(emp)
        print(f"  Dept: {emp.department.name:12} | Level: {level} | {emp.name}")


# =============================================================================
# Example 4: Multi-Level Sorting with Composite Keys
# =============================================================================

def example_composite_keys():
    """Demonstrate multi-level sorting using CompositeKeyExtractor."""
    print("\n" + "=" * 70)
    print("Example 4: Multi-Level Sorting with Composite Keys")
    print("=" * 70)
    
    # E-commerce products with multiple attributes
    products = [
        {
            "sku": "LAPTOP-001",
            "category": "Electronics",
            "subcategory": "Laptops",
            "pricing": {
                "base_price": 1200.00,
                "discount_percent": 10,
                "final_price": 1080.00
            },
            "inventory": {
                "warehouse": "US-West",
                "quantity": 50
            },
            "ratings": {
                "average": 4.5,
                "count": 120
            }
        },
        {
            "sku": "PHONE-001",
            "category": "Electronics",
            "subcategory": "Phones",
            "pricing": {
                "base_price": 800.00,
                "discount_percent": 15,
                "final_price": 680.00
            },
            "inventory": {
                "warehouse": "US-East",
                "quantity": 200
            },
            "ratings": {
                "average": 4.2,
                "count": 300
            }
        },
        {
            "sku": "LAPTOP-002",
            "category": "Electronics",
            "subcategory": "Laptops",
            "pricing": {
                "base_price": 1500.00,
                "discount_percent": 0,
                "final_price": 1500.00
            },
            "inventory": {
                "warehouse": "US-East",
                "quantity": 30
            },
            "ratings": {
                "average": 4.7,
                "count": 85
            }
        },
        {
            "sku": "TABLET-001",
            "category": "Electronics",
            "subcategory": "Tablets",
            "pricing": {
                "base_price": 600.00,
                "discount_percent": 20,
                "final_price": 480.00
            },
            "inventory": {
                "warehouse": "US-West",
                "quantity": 150
            },
            "ratings": {
                "average": 4.3,
                "count": 200
            }
        },
    ]
    
    print(f"\nProducts ({len(products)} items):")
    for p in products:
        print(f"  {p['sku']}: ${p['pricing']['final_price']:.2f} - {p['subcategory']}")
    
    # Multi-level sort: Category -> Subcategory -> Final Price
    print("\n--- Sorted by Category, Subcategory, Final Price ---")
    
    composite = CompositeKeyExtractor([
        lambda x: x["category"],
        lambda x: x["subcategory"],
        lambda x: x["pricing"]["final_price"],
    ])
    
    sorter = MadSorter(key_func=composite)
    sorted_products = sorter.sort(products.copy())
    
    for p in sorted_products:
        cat = p["category"]
        sub = p["subcategory"]
        price = p["pricing"]["final_price"]
        print(f"  {cat}/{sub}: ${price:>7.2f} - {p['sku']}")
    
    # Sort by warehouse, then by quantity (descending)
    print("\n--- Sorted by Warehouse, then Quantity (desc) ---")
    
    sorter = MadSorter(key_func=lambda x: (
        x["inventory"]["warehouse"],
        -x["inventory"]["quantity"]
    ))
    sorted_by_inventory = sorter.sort(products.copy())
    
    for p in sorted_by_inventory:
        wh = p["inventory"]["warehouse"]
        qty = p["inventory"]["quantity"]
        sku = p["sku"]
        print(f"  Warehouse: {wh} | Qty: {qty:3d} | {sku}")
    
    # Sort by rating (desc), then by review count (desc)
    print("\n--- Sorted by Rating, then Review Count ---")
    
    sorter = MadSorter(key_func=lambda x: (
        -x["ratings"]["average"],
        -x["ratings"]["count"]
    ))
    sorted_by_rating = sorter.sort(products.copy())
    
    for p in sorted_by_rating:
        rating = p["ratings"]["average"]
        count = p["ratings"]["count"]
        sku = p["sku"]
        print(f"  Rating: {rating:.1f} ({count:3d} reviews) | {sku}")


# =============================================================================
# Example 5: Real-World Log Entries with Nested Metadata
# =============================================================================

def example_log_entries():
    """Sort application logs with complex nested metadata."""
    print("\n" + "=" * 70)
    print("Example 5: Application Logs with Nested Metadata")
    print("=" * 70)
    
    log_entries = [
        {
            "timestamp": "2024-03-15T10:30:00Z",
            "level": "ERROR",
            "service": "payment-api",
            "message": "Payment processing failed",
            "metadata": {
                "request_id": "req-12345",
                "user_id": "user-789",
                "context": {
                    "endpoint": "/api/v1/pay",
                    "method": "POST",
                    "duration_ms": 5000,
                    "retry_count": 3
                },
                "error": {
                    "code": "TIMEOUT",
                    "details": "Gateway timeout"
                }
            }
        },
        {
            "timestamp": "2024-03-15T10:29:30Z",
            "level": "WARN",
            "service": "auth-service",
            "message": "Rate limit approaching",
            "metadata": {
                "request_id": "req-12344",
                "user_id": "user-456",
                "context": {
                    "endpoint": "/api/v1/auth",
                    "method": "GET",
                    "duration_ms": 150,
                    "retry_count": 0
                }
            }
        },
        {
            "timestamp": "2024-03-15T10:30:15Z",
            "level": "ERROR",
            "service": "payment-api",
            "message": "Invalid card number",
            "metadata": {
                "request_id": "req-12346",
                "user_id": "user-101",
                "context": {
                    "endpoint": "/api/v1/pay",
                    "method": "POST",
                    "duration_ms": 200,
                    "retry_count": 0
                },
                "error": {
                    "code": "VALIDATION",
                    "details": "Card validation failed"
                }
            }
        },
        {
            "timestamp": "2024-03-15T10:28:00Z",
            "level": "INFO",
            "service": "user-service",
            "message": "User profile updated",
            "metadata": {
                "request_id": "req-12343",
                "user_id": "user-789",
                "context": {
                    "endpoint": "/api/v1/user",
                    "method": "PUT",
                    "duration_ms": 80,
                    "retry_count": 0
                }
            }
        },
        {
            "timestamp": "2024-03-15T10:31:00Z",
            "level": "ERROR",
            "service": "database",
            "message": "Connection pool exhausted",
            "metadata": {
                "request_id": "req-12347",
                "user_id": "system",
                "context": {
                    "endpoint": "/internal/db",
                    "method": "QUERY",
                    "duration_ms": 10000,
                    "retry_count": 5
                },
                "error": {
                    "code": "DB_ERROR",
                    "details": "Max connections reached"
                }
            }
        },
    ]
    
    print(f"\nLog Entries ({len(log_entries)} total):")
    for log in log_entries:
        print(f"  [{log['level']:5}] {log['service']:15} | {log['message'][:30]}...")
    
    # Sort by severity (ERROR first), then by duration
    print("\n--- Sorted by Severity, then Duration ---")
    
    severity_order = {"ERROR": 0, "WARN": 1, "INFO": 2, "DEBUG": 3}
    
    sorter = MadSorter(key_func=lambda x: (
        severity_order.get(x["level"], 4),
        -x["metadata"]["context"]["duration_ms"]
    ))
    sorted_by_severity = sorter.sort(log_entries.copy())
    
    for log in sorted_by_severity:
        level = log["level"]
        duration = log["metadata"]["context"]["duration_ms"]
        service = log["service"]
        print(f"  [{level:5}] {duration:5}ms | {service}")
    
    # Sort by service, then by timestamp
    print("\n--- Sorted by Service, then Timestamp ---")
    
    sorter = MadSorter(key_func=lambda x: (x["service"], x["timestamp"]))
    sorted_by_service = sorter.sort(log_entries.copy())
    
    for log in sorted_by_service:
        ts = log["timestamp"].split("T")[1].rstrip("Z")
        service = log["service"]
        level = log["level"]
        print(f"  {ts} | {service:15} | [{level}]")
    
    # Sort by error code (group errors by type)
    print("\n--- Grouped by Error Code ---")
    
    def get_error_code(log):
        return log["metadata"].get("error", {}).get("code", "NO_ERROR")
    
    error_logs = [log for log in log_entries if "error" in log["metadata"]]
    sorter = MadSorter(key_func=get_error_code)
    sorted_by_error = sorter.sort(error_logs)
    
    for log in sorted_by_error:
        code = get_error_code(log)
        service = log["service"]
        duration = log["metadata"]["context"]["duration_ms"]
        print(f"  {code:15} | {service:15} | {duration}ms")
    
    # Sort by user ID, then by timestamp
    print("\n--- Sorted by User, then Time ---")
    
    sorter = MadSorter(key_func=lambda x: (x["metadata"]["user_id"], x["timestamp"]))
    sorted_by_user = sorter.sort(log_entries.copy())
    
    for log in sorted_by_user:
        user = log["metadata"]["user_id"]
        ts = log["timestamp"].split("T")[1].rstrip("Z")
        level = log["level"]
        print(f"  User: {user:10} | {ts} | [{level}] {log['message'][:25]}...")


# =============================================================================
# Bonus: Adaptive Sorting with Complex Objects
# =============================================================================

def example_adaptive_complex():
    """Demonstrate adaptive sorting with mixed complex objects."""
    print("\n" + "=" * 70)
    print("Bonus: Adaptive Sorting with Mixed Complex Objects")
    print("=" * 70)
    
    # Mixed data types
    mixed_data = [
        {"type": "user", "data": {"name": "Alice", "score": 95}},
        {"type": "product", "data": {"name": "Laptop", "price": 1200}},
        {"type": "user", "data": {"name": "Bob", "score": 87}},
        {"type": "order", "data": {"id": "ORD-001", "total": 2500}},
        {"type": "product", "data": {"name": "Phone", "price": 800}},
        {"type": "user", "data": {"name": "Charlie", "score": 92}},
    ]
    
    print(f"\nMixed Data ({len(mixed_data)} items):")
    for item in mixed_data:
        print(f"  [{item['type']:8}] {str(item['data'])[:40]}...")
    
    # Sort by type, then by name within each type
    print("\n--- Sorted by Type, then Name ---")
    
    def get_sort_key(item):
        type_val = item["type"]
        data = item["data"]
        # Extract name or other identifier
        name = data.get("name", data.get("id", "unknown"))
        return (type_val, name)
    
    sorter = AdaptiveMadSorter(
        initial_prefix_length=2,
        auto_adjust=True,
        enable_load_balance=True
    )
    
    sorted_mixed = sorter.sort(mixed_data.copy())
    
    for item in sorted_mixed:
        type_val = item["type"]
        data = item["data"]
        name = data.get("name", data.get("id", "unknown"))
        print(f"  [{type_val:8}] {name}")
    
    # Show adaptive report
    print("\n--- Adaptive Report ---")
    print(sorter.get_adaptive_report())


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    example_database_records()
    example_json_nested()
    example_parent_child()
    example_composite_keys()
    example_log_entries()
    example_adaptive_complex()
    
    print("\n" + "=" * 70)
    print("All examples completed!")
    print("=" * 70)
