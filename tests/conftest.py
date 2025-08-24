import os
import sys

HERE = os.path.dirname(__file__)
PIPE_DIR = os.path.abspath(os.path.join(HERE, ".."))
ROOT_DIR = os.path.abspath(os.path.join(PIPE_DIR, ".."))

for p in (PIPE_DIR, ROOT_DIR):
    if p not in sys.path:
        sys.path.insert(0, p)

