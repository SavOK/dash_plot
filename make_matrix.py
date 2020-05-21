from pathlib import Path
import itertools as itts
import csv, json

import pandas as pd
import numpy as np

dataLoc = Path("/home/saveliy/HHAlign/data")
inFile = dataLoc / "hhr_data_by_x_group_df.csv"

_PROJECT_LOC = Path("/home/saveliy/HHAlign/dash_plot")
_DATA_LOC = _PROJECT_LOC / "data_plots"

df = pd.read_csv(inFile, index_col=0)


def extract_matrix(
    df: pd.DataFrame,
    par: str,
    fill_val: float,
    val_undef: float = 0,
    sim_flag: bool = True,
) -> pd.DataFrame:
    """ returns matrix data frame from x by x data
    Arguments:
        df {pd.DataFrame} -- x by x data
        par {str} -- parameter 
        fill_val {float} -- fill value

    Keyword Arguments:
        val_undef {float} -- undefine value (default :{0})
        sim_flag {bool} -- is matrix symetric (default: {True})

    Returns:
        pd.DataFrame -- [description]
    """
    all_x_groups = sorted(set(list(df.x_template.unique()) + list(df.x_query.unique())))
    x_dict = {x: pos for pos, x in enumerate(all_x_groups)}
    matrix = np.full((len(x_dict), len(x_dict)), val_undef)
    if sim_flag:
        for ix1, gr1 in enumerate(all_x_groups):
            for ix2, gr2 in enumerate(all_x_groups[ix1:]):
                col = x_dict[gr1]
                row = x_dict[gr2]
                val = df[(df.x_template == gr1) & (df.x_query == gr2)][par]
                if val.empty:
                    matrix[row, col] = fill_val
                    matrix[col, row] = fill_val
                else:
                    matrix[row, col] = float(val)
                    matrix[col, row] = float(val)
    else:
        for ix1, gr1 in enumerate(all_x_groups):
            for ix2, gr2 in enumerate(all_x_groups):
                col = x_dict[gr1]
                row = x_dict[gr2]
                val = df[(df.x_template == gr1) & (df.x_query == gr2)][par]
                if val.empty:
                    matrix[row, col] = fill_val
                else:
                    matrix[row, col] = float(val)
    return pd.DataFrame(matrix, columns=all_x_groups, index=all_x_groups)


prob_matix = extract_matrix(df, "prob", 10.0)
prob_matix.to_csv(_DATA_LOC / "probability_matrix.txt")

score_matrix = extract_matrix(df, "score", 0, 0)
score_matrix.to_csv(_DATA_LOC / "score_matrix.txt")

