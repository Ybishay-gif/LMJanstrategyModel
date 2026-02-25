from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app import main

main(forced_view="tab2", multipage_mode=True)
