"""
Hash utilities for madS0rt.
Provides CRC32 and xxHash implementations with unified interface.
"""

import zlib
import binascii
from typing import Union, Protocol

try:
    import xxhash
    XXHASH_AVAILABLE = True
except ImportError:
    XXHASH_AVAILABLE = False


class HashProvider(Protocol):
    """Protocol for hash function providers."""
    
    def hash(self, data: Union[str, bytes], seed: int = 0) -> int:
        """Return hash as integer."""
        ...
    
    def hash_hex(self, data: Union[str, bytes], seed: int = 0) -> str:
        """Return hash as hexadecimal string."""
        ...


class CRC32Provider:
    """
    CRC32 hash provider (compatible with ptext.py implementation).
    Uses zlib.crc32 for consistent hashing.
    """
    
    def __init__(self):
        self.name = "crc32"
    
    def hash(self, data: Union[str, bytes], seed: int = 0) -> int:
        """
        Compute CRC32 hash as unsigned 32-bit integer.
        
        Args:
            data: String or bytes to hash
            seed: Optional seed value (ignored for CRC32)
        
        Returns:
            Unsigned 32-bit integer hash value
        """
        if isinstance(data, str):
            data = data.encode('utf-8')
        
        # Ensure positive 32-bit integer
        result = zlib.crc32(data) & 0xffffffff
        return result
    
    def hash_hex(self, data: Union[str, bytes], seed: int = 0) -> str:
        """Return CRC32 hash as 8-character hex string."""
        return format(self.hash(data, seed), '08x')


class XXHashProvider:
    """
    xxHash provider - faster alternative for large datasets.
    Falls back to CRC32 if xxhash not installed.
    """
    
    def __init__(self, bits: int = 32):
        self.name = f"xxhash{bits}"
        self.bits = bits
        self._available = XXHASH_AVAILABLE
        
        if not self._available:
            self._fallback = CRC32Provider()
    
    def hash(self, data: Union[str, bytes], seed: int = 0) -> int:
        """
        Compute xxHash as integer.
        
        Args:
            data: String or bytes to hash
            seed: Optional seed for randomized hashing
        
        Returns:
            Hash value as integer
        """
        if not self._available:
            return self._fallback.hash(data, seed)
        
        if isinstance(data, str):
            data = data.encode('utf-8')
        
        if self.bits == 32:
            return xxhash.xxh32(data, seed=seed).intdigest()
        elif self.bits == 64:
            return xxhash.xxh64(data, seed=seed).intdigest()
        else:
            raise ValueError(f"Unsupported xxHash bits: {self.bits}")
    
    def hash_hex(self, data: Union[str, bytes], seed: int = 0) -> str:
        """Return xxHash as hex string."""
        if not self._available:
            return self._fallback.hash_hex(data, seed)
        
        if isinstance(data, str):
            data = data.encode('utf-8')
        
        if self.bits == 32:
            return xxhash.xxh32(data, seed=seed).hexdigest()
        elif self.bits == 64:
            return xxhash.xxh64(data, seed=seed).hexdigest()
        else:
            raise ValueError(f"Unsupported xxHash bits: {self.bits}")


# Convenience functions matching ptext.py naming
def crc32_hash(data: Union[str, bytes]) -> int:
    """
    CRC32 hash function (ptext.py compatible).
    Returns unsigned 32-bit integer.
    """
    provider = CRC32Provider()
    return provider.hash(data)


def xxhash_hash(data: Union[str, bytes], bits: int = 32, seed: int = 0) -> int:
    """
    xxHash function with configurable bit width.
    
    Args:
        data: Data to hash
        bits: 32 or 64 bit hash
        seed: Optional seed value
    
    Returns:
        Hash as integer
    """
    provider = XXHashProvider(bits=bits)
    return provider.hash(data, seed)


def get_hash_provider(name: str = "crc32", **kwargs) -> HashProvider:
    """
    Factory function to get hash provider by name.
    
    Args:
        name: 'crc32' or 'xxhash32' or 'xxhash64'
        **kwargs: Additional provider-specific options
    
    Returns:
        HashProvider instance
    """
    name = name.lower()
    
    if name == "crc32":
        return CRC32Provider()
    elif name in ("xxhash", "xxhash32"):
        return XXHashProvider(bits=32)
    elif name == "xxhash64":
        return XXHashProvider(bits=64)
    else:
        raise ValueError(f"Unknown hash provider: {name}")
