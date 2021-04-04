import glob
import os.path
from segmentation import segmentation_ingress
from backend.utils.slack_messaging import slack_message
from pathlib import Path
import operator

full_path = "/media/jackson/967e3f65-43ea-4c63-bcdd-e124725e6d63/microscopy_images/"

file_paths = glob.iglob(full_path + '**/*.nd2', recursive=True)
files_and_sizes = ((path, os.path.getsize(path)) for path in file_paths)
sorted_files_with_size = sorted(files_and_sizes, key=operator.itemgetter(1))
for fp, _ in sorted_files_with_size:
    filename = Path(fp).name
    # slack_message(f'Starting Processing for {filename}...')
    segmentation_ingress(fp, 0, False)
    # slack_message(f'Finished Processing {filename}...')
