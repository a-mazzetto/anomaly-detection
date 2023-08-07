"""Utilities"""
import os
import gzip

def switched_open(filename, mode):
    """Open with gzip if necessary"""
    assert mode in ["r", "w"], "Unknown mode, must be 'r' or 'w'"
    ext = os.path.splitext(filename)[-1]
    if ext == ".gz":
        mode = "wt" if mode == "w" else "rt"
        return gzip.open(filename, mode, encoding="utf-8")
    elif ext in [".txt", ".json"]:
        return open(filename, mode, encoding="utf-8")
    else:
        raise ValueError("Unexpected file extension")
