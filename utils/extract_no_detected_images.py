import os.path
import shutil
from pathlib import Path

from src.utils.path_utils import iterate_recursively

base_path = ''
in_path = ['/home/malchul/work/GAN/StyleTransferDatasetPreparation/dumped/caricature_v2/good_samples']
out_path = '/home/malchul/work/utilites/pinterest-downloader/Caricature_merged'

all_files = [os.listdir(path) for path in in_path]
data_files = os.listdir()

out_path = Path(out_path)
out_path.mkdir(parents=True, exist_ok=True)
for i, img_path in enumerate(iterate_recursively(all_files)):
    base_name = os.path.basename(img_path)
    file_id = os.path.splitext(base_name)[0].split('_')[0]

    shutil.copyfile(img_path, out_path / f'{i}{os.path.splitext(img_path)[-1]}')