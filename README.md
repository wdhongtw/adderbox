# Personal Library and Toolbox

Contains general purpose library for Python developments.

All modules are provided under `adderbox` package, and they are intended to
be used separately. For example:

```python
from adderbox import control

# any resource that need to be cleaned up by a free function
resource = open("hello.txt")  
cleanup = lambda: resource.close()
with control.build_cm(resource, cleanup) as file:
    pass  # do something with file

from adderbox.folder import SkipList

# any iterable that need to be sorted
items = SkipList(["bob", "alice"])
assert next(iter(items)) == "alice"
```

This project always use inline generic variable.
So Python 3.12 and above is required.

It's guaranteed that there is no external dependency except standard library.

## Contents

- `control`: utilities around resource type and context managers
- `folder`: generic containers, e.g. ordered-mapping.
- `pipe`: pipeline style operation and related decorators

Just name a few. See src directory for more details.

## Development

Formatted with `black`.
