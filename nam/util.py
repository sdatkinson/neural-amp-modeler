# File: util.py
# Created Date: Sunday January 22nd 2023
# Author: Steven Atkinson (steven@atkinson.mn)

"""
Helpful utilities
"""

import os
import re
from pathlib import Path
from typing import List, Union, Optional
from datetime import datetime


def timestamp() -> str:
    t = datetime.now()
    return f"{t.year:04d}-{t.month:02d}-{t.day:02d}-{t.hour:02d}-{t.minute:02d}-{t.second:02d}"

def find_files(
    directory: str,
    extension: str,
    include_files: Optional[Union[str, List[str]]] = None,
    exclude_files: Optional[Union[str, List[str]]] = None) -> List[str]:
    """
    Returns a list of file paths in the given directory with the specified extension,
    and optionally filtered by included and excluded files.

    Args:
        directory (str): The directory to search for files.
        extension (str): The extension to search for (without the leading dot).
        include_files (Union[str, List[str]], optional): A regex or list of regex patterns for files to include
            in the search. Defaults to None, which includes all files.
        exclude_files (Union[str, List[str]], optional): A regex or list of regex patterns for files to exclude
            from the search. Defaults to None, which excludes no files.

    Returns:
        List[str]: A list of file paths of the matching files.

    Example:
        To search for all .wav files in a directory, use:
        >>> find_files('/path/to/directory', 'wav')
    """
    include_files = include_files.split(",") if isinstance(include_files, str) else include_files
    exclude_files = exclude_files.split(",") if isinstance(exclude_files, str) else exclude_files
    include_files = [] if include_files is None else include_files
    exclude_files = [] if exclude_files is None else exclude_files

    directory_path = Path(directory)
    files = []
    for file_path in directory_path.glob(f'*.{extension}'):
        if include_files:
            if not any(re.match(pattern, file_path.name) for pattern in include_files):
                continue
        if exclude_files:
            if any(re.match(pattern, file_path.name) for pattern in exclude_files):
                continue
        files.append(str(file_path))
    return sorted(files)
