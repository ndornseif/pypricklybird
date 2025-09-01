"""Convert binary data into a human friendly format.

[![PyPI - Version](https://img.shields.io/pypi/v/pypricklybird.svg)](https://pypi.org/project/pypricklybird)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/pypricklybird.svg)](https://pypi.org/project/pypricklybird)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
![GitHub License](https://img.shields.io/github/license/ndornseif/pypricklybird)

-----

## Overview

[`pricklybird`](https://github.com/ndornseif/pricklybird) is a method for conversion of
arbitrary binary data into more human-friendly words, <br>where each word represents
a single byte.<br>
A CRC-8 checksum is attached to allow the detection of errors during decoding.<br>
`0xDEADBEEF` becomes `turf-port-rust-warn-void`, for example.

`pypricklybird` is a python implementation of `pricklybird` version `v1`.

## Usage

Basic conversion functions that fully comply with the specification and
include the CRC can be used as follows.


>>> from pypricklybird import convert_to_pricklybird, convert_from_pricklybird
>>> data = bytearray.fromhex("4243")
>>> code = convert_to_pricklybird(data);
>>> code # Notice the third word "full" used to encode the CRC.
'flea-flux-full'
>>> convert_from_pricklybird(code).hex()
'4243'


It is also possible to map word to bytes and bytes to words without the
full standard implementation and CRC.
The words are encoded as four bytes of ASCII compatible UTF-8,
since the wordlist contains no non ASCII characters and all words are four letters long.

>>> from pypricklybird import words_to_bytes, bytes_to_words
>>> data = bytearray.fromhex("4243")
>>> words = bytes_to_words(data)
>>> words # Notice that no CRC is attached
['flea', 'flux']
>>> words_to_bytes(words).hex()
'4243'

Direct access to the `WORDLIST` used for mapping bytes to words,
and the `HASH_TABLE` use to map words to bytes is also possible.

>>> from pypricklybird import WORDLIST, HASH_TABLE, word_hash
>>> # Confirm that the word flux maps to the byte 0x43 in both directions.
>>> word = "flux"
>>> table_index = word_hash(word[0], word[-1]);
>>> table_index
603
>>> byte_value = HASH_TABLE[table_index]
>>> hex(byte_value)
'0x43'

## Installation

```console
pip install pypricklybird
```

## License

`pypricklybird` is distributed under the terms of the
[MIT](https://spdx.org/licenses/MIT.html) license.

## Documentation

Documentation is generated using [pdoc](https://pdoc.dev/) and can be found
[here](https://ndornseif.github.io/pypricklybird/).

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

__version__ = "1.0.2"

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
