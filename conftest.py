"""Makes src/ importable from the tests.

Every module in src/ imports its siblings flat (`from features import ...`),
because each one is also runnable as a script (`python src/train.py`). Putting
src/ on sys.path here keeps the tests importing the modules exactly the way the
scripts do, rather than through a package alias that would give `features` and
`src.features` two separate module objects with two separate copies of the cost
constants.
"""
import sys
from pathlib import Path

SRC = Path(__file__).parent / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
