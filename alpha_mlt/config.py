from pathlib import Path
import os

dataroot = os.getenv('ALPHA_MLT_DATAROOT', Path("~/alpha_mlt/trimmed").expanduser())
