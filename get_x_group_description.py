from pathlib import Path
import json

import pandas as pd

_MAIN_DIR = Path("/home/saveliy/HHAlign/dash_plot")
_DATA_LOC = _MAIN_DIR / "data_plots"


def get_x_names(inFile: Path = None):
    if inFile is None:
        fileLoc = Path("/home/saveliy/HHAlign/data")
        inFile = fileLoc / "ecod_desc.txt"
    df = pd.read_csv(inFile, sep="|")
    x_name_dict = {r["f_id"].split(".")[0]: r["x_name"] for ix, r in df.iterrows()}
    return x_name_dict


x_name_dict = get_x_names()
with open(_DATA_LOC / "x_group_names.json", "w") as oF:
    json.dump(x_name_dict, oF)
