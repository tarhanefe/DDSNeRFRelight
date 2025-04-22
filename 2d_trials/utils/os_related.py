import os
from typing import Optional

def find_latest_folder(parent_dir: str) -> Optional[str]:
    """
    Return the path to the most recently created sub‑directory of parent_dir.
    If there are no sub‑directories, returns None.
    """
    entries = (os.path.join(parent_dir, name) for name in os.listdir(parent_dir))
    dirs = [d for d in entries if os.path.isdir(d)]
    if not dirs:
        return None
    return max(dirs, key=os.path.getctime)

def find_first_final_file(folder: str, keyword: str = "final") -> Optional[str]:
    """
    Return the full path of the first file in `folder` whose name contains `keyword` (case‑insensitive).
    If none found, returns None.
    """
    for fname in os.listdir(folder):
        full = os.path.join(folder, fname)
        if os.path.isfile(full) and keyword.lower() in fname.lower():
            return full
    return None

def find_final_in_latest(parent_dir: str) -> Optional[str]:
    """
    Find the most recently created sub‑folder of `parent_dir`, then
    return the path to the first file in it containing "final".
    """
    latest = find_latest_folder(parent_dir)
    if not latest:
        return None
    return find_first_final_file(latest)