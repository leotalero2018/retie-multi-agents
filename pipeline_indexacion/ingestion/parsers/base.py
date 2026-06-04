from pathlib import Path
from typing import List, Tuple

class BaseParser:
    def extract_pages(self, path: Path) -> List[Tuple[int, str]]:
        raise NotImplementedError
