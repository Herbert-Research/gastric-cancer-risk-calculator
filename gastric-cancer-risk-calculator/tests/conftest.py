import sys
from pathlib import Path

# Add the root directory to sys.path so models can be imported
root_dir = Path(__file__).parent.parent.parent
sys.path.insert(0, str(root_dir))
