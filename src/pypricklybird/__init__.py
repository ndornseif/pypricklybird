"""Convert binary data into a human friendly format.

This is a python implementation of the [pricklybird][1] format `v1`.

## Usage
```python
>>> from pypricklybird import convert_to_pricklybird, convert_from_pricklybird
>>> data = bytearray.fromhex("4243")
>>> code = convert_to_pricklybird(data);
>>> code
'flea-flux-full'
>>> convert_from_pricklybird(code).hex()
'4243'

```

## Documentation

[1]: https://github.com/ndornseif/pricklybird
"""

from pypricklybird.converter import (
    CRC8_POLY,
    CRC8_TABLE,
    HASH_TABLE,
    HASH_TABLE_SIZE,
    WORDLIST,
    CRCError,
    DecodeError,
    bytes_to_words,
    calculate_crc8,
    convert_from_pricklybird,
    convert_to_pricklybird,
    word_hash,
    words_to_bytes,
)

__version__ = "1.0.1"

PRICKLYBIRD_VERSION = "v1"
"""Version of the pricklybird specification that this implementation complies with."""

__all__ = [
    "CRC8_POLY",
    "CRC8_TABLE",
    "HASH_TABLE",
    "HASH_TABLE_SIZE",
    "PRICKLYBIRD_VERSION",
    "WORDLIST",
    "CRCError",
    "DecodeError",
    "bytes_to_words",
    "calculate_crc8",
    "convert_from_pricklybird",
    "convert_to_pricklybird",
    "word_hash",
    "words_to_bytes",
]
