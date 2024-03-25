"""
    Utilitiy functions
"""

# pylint: disable=C0301,C0103,C0303,C0411,W1203

import os
import shutil

def recreate_dirs(dir_list : list[str]) -> None:
    """
        Delete and create folders
    """
    for dir_path in dir_list:
        try:
            if os.path.isdir(dir_path):
                shutil.rmtree(dir_path)
        except: # pylint: disable=W0702
            pass        
        try:
            os.makedirs(dir_path, exist_ok=True)
        except: # pylint: disable=W0702
            pass
        
def save_text_file_utf8(file_path: str, data: str) -> None:
    """
        Save text data to file as UTF-8
    """
    with open(file_path, 'w', encoding = 'utf-8') as f:
        f.write(data)
    