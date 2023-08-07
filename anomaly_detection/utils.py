"""Utilities"""
import os
import gzip
import subprocess

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

def switched_system_file_sort(filename):
    """Function to sort inplace"""
    if os.name == "nt":
        completed = subprocess.run(["powershell", "-Command",
            f"Get-Content {output_file} | Sort-Object | Set-Content -Path {output_file}"],
            capture_output=True)
        if completed.returncode != 0:
            raise completed.stderr
    else:
        if os.path.splitext(filename)[-1] == ".gz":
            command = f"zcat {filename} | sort -k 1 | gzip > temp.gz; cp -f temp.gz {filename}; rm temp.gz"
        else:
            command = f"sort -o {filename} -k 1 {filename}"
        completed = os.system(command=command)
        if completed != 0:
            raise RuntimeError("Issues sorting file")
    print(f"Sorting of {filename} executed successfully!")
