# pypricklybird

[![PyPI - Version](https://img.shields.io/pypi/v/pypricklybird.svg)](https://pypi.org/project/pypricklybird)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/pypricklybird.svg)](https://pypi.org/project/pypricklybird)

-----

## Overview 
This is a python implementation of the [pricklybird](https://github.com/ndornseif/pricklybird) format version `v1`.

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
Documentation is generated using [pdoc](https://pdoc.dev/) and can be found [here](https://ndornseif.github.io/pypricklybird/).

## Installation

```console
pip install pypricklybird
```

## License

`pypricklybird` is distributed under the terms of the [MIT](https://spdx.org/licenses/MIT.html) license.
