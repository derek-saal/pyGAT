from pathlib import Path
from sys import path

proj_dir = Path(__file__).parents[2]
data_dir = proj_dir/'mr_data'

path.append(str(proj_dir))
