import sys
from pathlib import Path

# Make `import src...` work regardless of where pytest is invoked from.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
