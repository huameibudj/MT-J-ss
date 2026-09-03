import sys
sys.path.append(r'E:\GitRepo\Pinglib')  # 加入你的项目根目录到 sys.path
from pinglab_benchmarks.utils import unpack_segmentation_lmdb

lmdb_path = r'B:\Benchmarks\Segmentation\[5] ICC-segmentation-Jiawei-4class.lmdb'
target_folder = r'D:\lhxworkspace\data\ICCSegmentation'

unpack_segmentation_lmdb(lmdb_path, target_folder)
