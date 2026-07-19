"""
Key extraction utilities for madS0rt.
Provides various strategies for extracting sort keys from complex objects.
Supports string, numeric, and composite keys.
"""

from typing import Any, Callable, List, Optional, Union, Pattern, Tuple
from abc import ABC, abstractmethod
import re


class BaseExtractor(ABC):
    """Abstract base class for all extractors."""
    
    @abstractmethod
    def extract(self, item: Any) -> Union[str, int, float, Tuple]:
        """Extract key from item. Returns string, number, or tuple."""
        ...
    
    def __call__(self, item: Any) -> Union[str, int, float, Tuple]:
        """Make extractor callable."""
        return self.extract(item)


class FirstNCharsExtractor(BaseExtractor):
    """
    Extracts first N characters from string representation.
    Enhanced version with padding, case options, and normalization.
    """
    
    def __init__(
        self,
        n: int = 3,
        lowercase: bool = True,
        normalize: bool = True,
        pad: Optional[str] = None,
        pad_length: Optional[int] = None,
        from_end: bool = False
    ):
        """
        Initialize FirstNCharsExtractor.
        
        Args:
            n: Number of characters to extract
            lowercase: Convert to lowercase
            normalize: Remove accents using unidecode
            pad: Character to pad with if string shorter than n
            pad_length: Total length after padding (defaults to n)
            from_end: Extract from end instead of beginning
        """
        self.n = n
        self.lowercase = lowercase
        self.normalize = normalize
        self.pad = pad
        self.pad_length = pad_length or n
        self.from_end = from_end
    
    def extract(self, item: Any) -> str:
        """Extract first N characters."""
        key = self._to_string(item)
        
        if self.lowercase:
            key = key.lower()
        
        if self.normalize:
            key = self._normalize(key)
        
        # Extract substring
        if self.from_end:
            result = key[-self.n:] if len(key) >= self.n else key
        else:
            result = key[:self.n] if len(key) >= self.n else key
        
        # Pad if necessary
        if self.pad and len(result) < self.pad_length:
            result = result + (self.pad * (self.pad_length - len(result)))
        
        return result
    
    def _to_string(self, item: Any) -> str:
        """Convert item to string."""
        if isinstance(item, str):
            return item
        elif hasattr(item, '__str__'):
            return str(item)
        else:
            return repr(item)
    
    def _normalize(self, s: str) -> str:
        """Normalize string by removing accents."""
        try:
            from unidecode import unidecode
            return unidecode(s)
        except ImportError:
            return s
    
    def set_n(self, n: int) -> 'FirstNCharsExtractor':
        """Chainable method to update N."""
        self.n = n
        return self


class LastNCharsExtractor(BaseExtractor):
    """Extracts last N characters (useful for file extensions, suffixes)."""
    
    def __init__(self, n: int = 3, **kwargs):
        super().__init__()
        self._extractor = FirstNCharsExtractor(n=n, from_end=True, **kwargs)
    
    def extract(self, item: Any) -> str:
        return self._extractor.extract(item)


class CustomRegexExtractor(BaseExtractor):
    """
    Extracts key using regex pattern matching.
    Supports multiple match groups and custom transformations.
    """
    
    def __init__(
        self,
        pattern: Union[str, Pattern],
        group: Union[int, str] = 0,
        fallback: Any = "",
        transform: Optional[Callable[[str], Any]] = None,
        multiple: bool = False,
        separator: str = "|"
    ):
        """
        Initialize CustomRegexExtractor.
        
        Args:
            pattern: Regex pattern string or compiled pattern
            group: Capture group index or name to return
            fallback: Value to return if no match
            transform: Optional function to transform match result
            multiple: If True, return all matches joined by separator
            separator: Separator for multiple matches
        """
        self.pattern = re.compile(pattern) if isinstance(pattern, str) else pattern
        self.group = group
        self.fallback = fallback
        self.transform = transform
        self.multiple = multiple
        self.separator = separator
    
    def extract(self, item: Any) -> Union[str, List[str], Any]:
        """Extract using regex pattern."""
        text = str(item)
        
        if self.multiple:
            matches = self.pattern.findall(text)
            if not matches:
                return self.fallback
            
            # Handle tuple matches from named groups
            processed = []
            for m in matches:
                if isinstance(m, tuple):
                    processed.append(m[0] if m else "")
                else:
                    processed.append(str(m))
            
            result = self.separator.join(processed)
        else:
            match = self.pattern.search(text)
            if match:
                try:
                    result = match.group(self.group)
                except (IndexError, KeyError):
                    result = self.fallback
            else:
                result = self.fallback
        
        if self.transform and result != self.fallback:
            result = self.transform(result)
        
        return result


class NumericExtractor(BaseExtractor):
    """
    Extracts numeric values from strings or objects.
    Supports integers, floats, and custom number formats.
    """
    
    def __init__(
        self,
        pattern: Optional[str] = None,
        group: int = 1,
        as_type: type = int,
        default: Any = 0,
        allow_negative: bool = True,
        allow_decimal: bool = True
    ):
        """
        Initialize NumericExtractor.
        
        Args:
            pattern: Custom regex pattern (default: auto-detect numbers)
            group: Capture group containing the number
            as_type: Convert to int or float
            default: Default value if no number found
            allow_negative: Include minus sign
            allow_decimal: Include decimal point
        """
        if pattern:
            self.pattern = re.compile(pattern)
        else:
            # Build pattern based on options
            sign = r"-?" if allow_negative else r""
            decimal = r"(?:\.\d+)?" if allow_decimal else r""
            self.pattern = re.compile(rf".*?({sign}\d+{decimal})")
        
        self.group = group
        self.as_type = as_type
        self.default = default
    
    def extract(self, item: Any) -> Union[int, float]:
        """Extract numeric value."""
        text = str(item)
        match = self.pattern.search(text)
        
        if match:
            try:
                num_str = match.group(self.group)
                return self.as_type(num_str)
            except (ValueError, IndexError):
                pass
        
        return self.default


class MultiFieldExtractor(BaseExtractor):
    """
    Extracts and combines multiple fields from dicts or objects.
    Supports different extraction strategies per field.
    """
    
    def __init__(
        self,
        fields: List[Union[str, Tuple[str, BaseExtractor]]],
        separator: str = "|",
        missing: str = "_",
        extractor: Optional[BaseExtractor] = None
    ):
        """
        Initialize MultiFieldExtractor.
        
        Args:
            fields: List of field names or (name, extractor) tuples
            separator: String to join field values
            missing: Value for missing fields
            extractor: Default extractor to apply to each field
        """
        self.fields = fields
        self.separator = separator
        self.missing = missing
        self.default_extractor = extractor or FirstNCharsExtractor(n=100, normalize=False)
    
    def extract(self, item: Any) -> str:
        """Extract multiple fields and combine."""
        parts = []
        
        for field_spec in self.fields:
            if isinstance(field_spec, tuple):
                field_name, extractor = field_spec
            else:
                field_name, extractor = field_spec, self.default_extractor
            
            value = self._get_field(item, field_name)
            if value is None:
                parts.append(self.missing)
            else:
                extracted = extractor.extract(value) if hasattr(extractor, 'extract') else extractor(value)
                parts.append(str(extracted))
        
        return self.separator.join(parts)
    
    def _get_field(self, item: Any, field: str) -> Any:
        """Get field value from dict or object."""
        if isinstance(item, dict):
            return item.get(field)
        elif hasattr(item, field):
            return getattr(item, field)
        else:
            # Try nested attribute access (e.g., "user.name")
            parts = field.split('.')
            current = item
            for part in parts:
                if hasattr(current, part):
                    current = getattr(current, part)
                elif isinstance(current, dict) and part in current:
                    current = current[part]
                else:
                    return None
            return current
    
    def extract_tuple(self, item: Any) -> Tuple:
        """Extract as tuple instead of joined string (for multi-key sorting)."""
        parts = []
        
        for field_spec in self.fields:
            if isinstance(field_spec, tuple):
                field_name, extractor = field_spec
            else:
                field_name, extractor = field_spec, self.default_extractor
            
            value = self._get_field(item, field_name)
            if value is None:
                parts.append(self.missing)
            else:
                extracted = extractor.extract(value) if hasattr(extractor, 'extract') else extractor(value)
                parts.append(extracted)
        
        return tuple(parts)


class CompositeKeyExtractor(BaseExtractor):
    """
    Creates composite keys for multi-level sorting.
    Returns tuple for use with Python's tuple comparison.
    """
    
    def __init__(
        self,
        extractors: List[BaseExtractor],
        priorities: Optional[List[int]] = None
    ):
        """
        Initialize CompositeKeyExtractor.
        
        Args:
            extractors: List of extractors to combine
            priorities: Optional priority weights for each extractor
        """
        self.extractors = extractors
        self.priorities = priorities or list(range(len(extractors)))
    
    def extract(self, item: Any) -> Tuple:
        """Extract composite key as tuple."""
        values = []
        for ext in self.extractors:
            val = ext.extract(item) if hasattr(ext, 'extract') else ext(item)
            values.append(val)
        return tuple(values)


class ConditionalExtractor(BaseExtractor):
    """
    Applies different extractors based on conditions.
    """
    
    def __init__(
        self,
        conditions: List[Tuple[Callable[[Any], bool], BaseExtractor]],
        default: Optional[BaseExtractor] = None
    ):
        """
        Initialize ConditionalExtractor.
        
        Args:
            conditions: List of (predicate, extractor) tuples
            default: Extractor to use if no condition matches
        """
        self.conditions = conditions
        self.default = default or FirstNCharsExtractor(n=3)
    
    def extract(self, item: Any) -> Any:
        """Apply first matching extractor."""
        for predicate, extractor in self.conditions:
            if predicate(item):
                return extractor.extract(item) if hasattr(extractor, 'extract') else extractor(item)
        return self.default.extract(item) if hasattr(self.default, 'extract') else self.default(item)


class ChainExtractor(BaseExtractor):
    """
    Chains multiple extractors (output of one is input to next).
    """
    
    def __init__(self, extractors: List[BaseExtractor]):
        self.extractors = extractors
    
    def extract(self, item: Any) -> Any:
        """Chain extractors."""
        result = item
        for ext in self.extractors:
            result = ext.extract(result) if hasattr(ext, 'extract') else ext(result)
        return result


# Backwards compatibility aliases
PrefixExtractor = FirstNCharsExtractor
RegexExtractor = CustomRegexExtractor


# Factory functions for common use cases
def make_extractor(type_name: str, **kwargs) -> BaseExtractor:
    """
    Factory function to create extractors by type name.
    
    Args:
        type_name: One of 'prefix', 'suffix', 'regex', 'numeric', 
                   'multi_field', 'composite', 'conditional', 'chain'
        **kwargs: Constructor arguments for the extractor
    
    Returns:
        Configured extractor instance
    """
    extractors = {
        'prefix': FirstNCharsExtractor,
        'suffix': LastNCharsExtractor,
        'regex': CustomRegexExtractor,
        'numeric': NumericExtractor,
        'multi_field': MultiFieldExtractor,
        'composite': CompositeKeyExtractor,
        'conditional': ConditionalExtractor,
        'chain': ChainExtractor,
    }
    
    if type_name not in extractors:
        raise ValueError(f"Unknown extractor type: {type_name}")
    
    return extractors[type_name](**kwargs)


# Convenience presets
def make_filename_extractor() -> MultiFieldExtractor:
    """Extractor optimized for filename sorting (name + extension)."""
    return MultiFieldExtractor(
        fields=[
            ('name', CustomRegexExtractor(r'^(.*?)(?:\.[^.]+)?$', group=1)),
            ('ext', CustomRegexExtractor(r'\.([^.]+)$$', group=1, fallback='')),
        ],
        separator='.'
    )


def make_version_extractor() -> CompositeKeyExtractor:
    """Extractor for version strings (e.g., 1.2.3 -> (1, 2, 3))."""
    return CompositeKeyExtractor([
        NumericExtractor(pattern=r'(\d+)', group=1, default=0)
        for _ in range(4)  # Support up to 4 version components
    ])


def make_date_extractor(pattern: str = r'(\d{4})-(\d{2})-(\d{2})') -> CompositeKeyExtractor:
    """Extractor for dates (year, month, day)."""
    return CompositeKeyExtractor([
        NumericExtractor(pattern=pattern, group=i, default=0)
        for i in range(1, 4)
    ])
