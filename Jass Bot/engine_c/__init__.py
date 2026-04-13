"""C-accelerated Jass engine."""

import os
import sys

if sys.platform == "win32":
    _ucrt_bin = "C:/msys64/ucrt64/bin"
    if os.path.isdir(_ucrt_bin):
        os.add_dll_directory(_ucrt_bin)
