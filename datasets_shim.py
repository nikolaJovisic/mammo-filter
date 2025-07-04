import sys
from pathlib import Path

repos_dir = "/home/nikola.jovisic.ivi/nj/"

sys.path.append(str(Path(repos_dir)))
sys.path.append(str(Path(repos_dir + "mammo_datasets")))

from mammo_datasets import *