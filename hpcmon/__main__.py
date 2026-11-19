"""Allow running hpcmon as ``python -m hpcmon``."""

import sys

from hpcmon.cli import main

if __name__ == "__main__":
    sys.exit(main())
