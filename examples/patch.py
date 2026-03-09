from pathlib import Path
import sys

repo_root = Path.cwd()
if not (repo_root / "release").exists():
    repo_root = repo_root.parent
sys.path.insert(0, str(repo_root))
