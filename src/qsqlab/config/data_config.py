from pathlib import Path
from platformdirs import user_data_dir

# This global variable defines the default folder in which the python library stores the measurement 
# data independently of operating system.
DEFAULT_DATA_ROOT: Path = Path(user_data_dir("qsqlab")) / "data_dump"