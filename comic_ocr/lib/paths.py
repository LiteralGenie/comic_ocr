from pathlib import Path


SRC_DIR = Path(__file__).parent.parent
PROJ_DIR = SRC_DIR.parent
CACHE_DIR = PROJ_DIR / "cache"

for dir in [CACHE_DIR]:
    dir.mkdir(exist_ok=True)
