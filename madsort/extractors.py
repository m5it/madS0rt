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
        self.n = n
        self.lowercase = lowercase
        self.normalize = normalize
        self.pad = pad
        self.pad_length = pad_length or n
        self.from_end = from_end
    
    def extract(self, item: Any) -> str:
        key = self._to_string(item)
        if self.lowercase:
            key = key.lower()
        if self.normalize:
            key = self._normalize(key)
        
        if self.from_end:
            result = key[-self.n:] if len(key) >= self.n else key
        else:
            result = key[:self.n] if len(key) >= self.n else key
        
        if self.pad and len(result) < self.pad_length:
            result = result + (self.pad * (self.pad_length - len(result)))
        
        return result
    
    def _to_string(self, item: Any) -> str:
        if isinstance(item, str):
            return item
        elif hasattr(item, '__str__'):
            return str(item)
        else:
            return repr(item)
    
    def _normalize(self, s: str) -> str:
        try:
            from unidecode import unidecode
            return unidecode(s)
        except ImportError:
            return s


class LastNCharsExtractor(BaseExtractor):
    def __init__(self, n: int = 3, **kwargs):
        super().__init__()
        self._extractor = FirstNCharsExtractor(n=n, from_end=True, **kwargs)
    
    def extract(self, item: Any) -> str:
        return self._extractor.extract(item)


class CustomRegexExtractor(BaseExtractor):
    def __init__(
        self,
        pattern: Union[str, Pattern],
        group: Union[int, str] = 0,
        fallback: Any = "",
        transform: Optional[Callable[[str], Any]] = None,
        multiple: bool = False,
        separator: str = "|"
    ):
        self.pattern = re.compile(pattern) if isinstance(pattern, str) else pattern
        self.group = group
        self.fallback = fallback
        self.transform = transform
        self.multiple = multiple
        self.separator = separator
    
    def extract(self, item: Any) -> Union[str, List[str], Any]:
        text = str(item)
        if self.multiple:
            matches = self.pattern.findall(text)
            if not matches:
                return self.fallback
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
    def __init__(
        self,
        pattern: Optional[str] = None,
        group: int = 1,
        as_type: type = int,
        default: Any = 0,
        allow_negative: bool = True,
        allow_decimal: bool = True
    ):
        if pattern:
            self.pattern = re.compile(pattern)
        else:
            sign = r"-?" if allow_negative else r""
            decimal = r"(?:\.\d+)?" if allow_decimal else r""
            self.pattern = re.compile(rf".*?({sign}\d+{decimal})")
        
        self.group = group
        self.as_type = as_type
        self.default = default
    
    def extract(self, item: Any) -> Union[int, float]:
        text = str(item)
        match = self.pattern.search(text)
        if match:
            try:
                num_str = match.group(self.group)
                return self.as_type(num_str)
            except (ValueError, IndexError):
                pass
        return self.default


class PathExtractor(BaseExtractor):
    """
    Extracts values using deep path access like 'user.profile.name' 
    or ['user']['profile']['name'] for nested objects and dicts.
    """
    
    def __init__(
        self,
        path: Union[str, List[str]],
        separator: str = ".",
        default: Any = None
    ):
        """
        Initialize PathExtractor.
        
        Args:
            path: Dot-separated string or list of keys for deep access
            separator: Separator for path string (default: '.')
            default: Default value if path not found
        """
        if isinstance(path, str):
            self.path_parts = path.split(separator)
        else:
            self.path_parts = list(path)
        self.separator = separator
        self.default = default
    
    def extract(self, item: Any) -> Any:
        """
        Extract value at path from item.
        Supports both dict access (data['key']) and attribute access (obj.attr).
        """
        current = item
        for part in self.path_parts:
            if current is None:
                return self.default
            
            # Try dict access first
            if isinstance(current, dict):
                if part in current:
                    current = current[part]
                else:
                    return self.default
            # Then try attribute access
            elif hasattr(current, part):
                current = getattr(current, part)
            else:
                return self.default
        
        return current
    
    def __repr__(self) -> str:
        return f"PathExtractor({'/'.join(self.path_parts)})"


class FieldExtractor(BaseExtractor):
    """
    Extracts field from dict or object attribute.
    Enhanced to support nested paths like 'user.profile.name'.
    """
    
    def __init__(
        self,
        field: str,
        default: str = "",
        separator: str = "."
    ):
        """
        Initialize FieldExtractor.
        
        Args:
            field: Field name or nested path (e.g., 'user.name' or 'name')
            default: Default value if field not found
            separator: Path separator for nested access (default: '.')
        """
        self.field = field
        self.default = default
        self.separator = separator
        self._path_extractor = PathExtractor(field, separator, default)
    
    def extract(self, item: Any) -> str:
        """Extract field from item using path."""
        result = self._path_extractor.extract(item)
        return str(result) if result is not None else self.default


class MultiFieldExtractor(BaseExtractor):
    """
    Extracts and combines multiple fields from dicts or objects.
    Enhanced to support nested paths.
    """
    
    def __init__(
        self,
        fields: List[Union[str, Tuple[str, BaseExtractor]]],
        separator: str = "|",
        missing: str = "_",
        extractor: Optional[BaseExtractor] = None
    ):
        self.fields = fields
        self.separator = separator
        self.missing = missing
        self.default_extractor = extractor or FieldExtractor(field="", default=missing)
    
    def extract(self, item: Any) -> str:
        """Extract multiple fields and combine."""
        parts = []
        
        for field_spec in self.fields:
            if isinstance(field_spec, tuple):
                field_name, extractor = field_spec
            else:
                field_name, extractor = field_spec, self.default_extractor
            
            # Use FieldExtractor for string paths, otherwise use as-is
            if isinstance(field_name, str) and isinstance(extractor, FieldExtractor):
                value = extractor.extract(item)
            elif isinstance(extractor, PathExtractor):
                value = str(extractor.extract(item))
            else:
                # Try direct access first
                value = self._get_field(item, field_name)
                if value is not None and hasattr(extractor, 'extract'):
                    extracted = extractor.extract(value)
                    value = str(extracted) if extracted is not None else self.missing
                elif value is None:
                    value = self.missing
            
            parts.append(str(value))
        
        return self.separator.join(parts)
    
    def _get_field(self, item: Any, field: str) -> Any:
        """Get field value using PathExtractor."""
        path_ext = PathExtractor(field, self.separator, None)
        return path_ext.extract(item)
    
    def extract_tuple(self, item: Any) -> Tuple:
        """Extract as tuple for multi-key sorting."""
        parts = []
        
        for field_spec in self.fields:
            if isinstance(field_spec, tuple):
                field_name, extractor = field_spec
            else:
                field_name, extractor = field_spec, self.default_extractor
            
            if isinstance(extractor, (FieldExtractor, PathExtractor)):
                value = extractor.extract(item)
            else:
                value = self._get_field(item, field_name)
                if value is not None and hasattr(extractor, 'extract'):
                    value = extractor.extract(value)
            
            parts.append(value if value is not None else self.missing)
        
        return tuple(parts)


class CompositeKeyExtractor(BaseExtractor):
    """Creates composite keys for multi-level sorting."""
    
    def __init__(
        self,
        extractors: List[BaseExtractor],
        priorities: Optional[List[int]] = None
    ):
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
    """Applies different extractors based on conditions."""
    
    def __init__(
        self,
        conditions: List[Tuple[Callable[[Any], bool], BaseExtractor]],
        default: Optional[BaseExtractor] = None
    ):
        self.conditions = conditions
        self.default = default or FirstNCharsExtractor(n=3)
    
    def extract(self, item: Any) -> Any:
        """Apply first matching extractor."""
        for predicate, extractor in self.conditions:
            if predicate(item):
                return extractor.extract(item) if hasattr(extractor, 'extract') else extractor(item)
        return self.default.extract(item) if hasattr(self.default, 'extract') else self.default(item)


class ChainExtractor(BaseExtractor):
    """Chains multiple extractors (output of one is input to next)."""
    
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


# Factory functions
def make_extractor(type_name: str, **kwargs) -> BaseExtractor:
    """Factory function to create extractors by type name."""
    extractors = {
        'prefix': FirstNCharsExtractor,
        'suffix': LastNCharsExtractor,
        'regex': CustomRegexExtractor,
        'numeric': NumericExtractor,
        'path': PathExtractor,
        'field': FieldExtractor,
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
    """Extractor optimized for filename sorting."""
    return MultiFieldExtractor(
        fields=[
            ('name', CustomRegexExtractor(r'^(.*?)(?:\.[^.]+)?$', group=1)),
            ('ext', CustomRegexExtractor(r'\.([^.]+)$$', group=1, fallback='')),
        ],
        separator='.'
    )


def make_version_extractor() -> CompositeKeyExtractor:
    """Extractor for version strings."""
    return CompositeKeyExtractor([
        NumericExtractor(pattern=r'(\d+)', group=1, default=0)
        for _ in range(4)
    ])


def make_date_extractor(pattern: str = r'(\d{4})-(\d{2})-(\d{2})') -> CompositeKeyExtractor:
    """Extractor for dates."""
    return CompositeKeyExtractor([
        NumericExtractor(pattern=pattern, group=i, default=0)
        for i in range(1, 4)
    ])
