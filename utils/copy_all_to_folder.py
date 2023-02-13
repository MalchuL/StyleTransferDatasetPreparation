import os.path
import shutil
from pathlib import Path

from src.utils.path_utils import iterate_recursively

in_path = '/home/malchul/work/utilites/pinterest-downloader/Caricature'
out_path = '/home/malchul/work/utilites/pinterest-downloader/Caricature_merged'

out_path = Path(out_path)
out_path.mkdir(parents=True, exist_ok=True)
for i, img_path in enumerate(iterate_recursively(in_path)):
    shutil.copyfile(img_path, out_path / f'{i}{os.path.splitext(img_path)[-1]}')