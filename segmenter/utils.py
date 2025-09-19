import glob
import os


def check_path_exists(path: str, pattern: str, clear: bool = False):
    """
    Check if directory path exists
    if clear is True then delete the contents
    """
    if os.path.exists(path):
        if clear:
            for f in glob.glob(pattern):
                os.remove(f)
    else:
        os.makedirs(path)

