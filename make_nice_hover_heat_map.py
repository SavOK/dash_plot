from pathlib import Path
import json
import pickle
from typing import List, Dict, String, Integer


import pandas as pd
import numpy as np

import plotly.graph_objects as go

import dash
import dash_core_components as dcc
import dash_html_components as html

_MAIN_DIR = Path("C:/Users/Saveliy/Projects/HHAlign/dash_plot")
_DATA_LOC = _MAIN_DIR / "data_plots"

_PROB_MATRIX = _DATA_LOC / "probability_matrix.txt"


def read_x_names(inFile: Path = _DATA_LOC / "x_group_names.json") -> Dict:
    with open(inFile) as oF:
        return json.load(oF)


data = pd.read_csv(_PROB_MATRIX, index_col=0)
x_names = read_x_names()


def _nice_lables(x, cut: Integer = 20, name_dict: Dict = x_names) -> String:
    name = name_dict[str(x)]
    if len(name) > cut:
        return f"{name[:cut-3]}..."
    else:
        return name


# def _make_hover_over(df_plot: pd.DataFrame, df_x_group: pd.DataFrame,
#                      x_range: Integer, y_range: Integer) -> Dict:
#     pass


def _make_cs_data(
    df_plot, df_x_group, x_range: int = None, y_range: int = None
) -> List[List]:
    if x_range is None:
        x_range = len(df_plot.columns)
    if y_range is None:
        y_range = len(df_plot.index)

    cs_data = [[""] * y_range for x in range(x_range)]
    for j, J in enumerate(df_plot.index[:y_range]):
        for i, I in enumerate(df_plot.columns[:x_range]):
            if int(I) < int(J):
                curr_line = df_x_group[
                    (df_x_group["x_template"] == int(I))
                    & (df_x_group["x_query"] == int(J))
                ]
            else:
                curr_line = df_x_group[
                    (df_x_group["x_template"] == int(J))
                    & (df_x_group["x_query"] == int(I))
                ]
            str_temp = "X-group {}: {}<br>".format(I, _nice_lables(I, 30))
            str_temp += "X-group {}: {}<br>".format(J, _nice_lables(J, 30))

            if curr_line.empty:
                str_temp += "<extra></extra>"
                cs_data[j][i] = str_temp
            else:
                if int(I) < int(J):
                    str_temp += "domain 1: {}<br>".format(
                        curr_line["template"].values[0]
                    )
                    str_temp += "domain 2: {}<br>".format(
                        curr_line["query"].values[0])
                else:
                    str_temp += "domain 1: {}<br>".format(
                        curr_line["query"].values[0])
                    str_temp += "domain 2: {}<br>".format(
                        curr_line["template"].values[0]
                    )
                str_temp += "HH-Probability: {:.2f}<br>".format(
                    curr_line.prob.values[0]
                )
                str_temp += "Score: {:.2f}<br>".format(
                    curr_line.score.values[0])
                str_temp += "Align Length: {}<br>".format(
                    int(curr_line.Cols.values[0]))
                str_temp += "Align Ratio: {:.2f}<br>".format(
                    curr_line.align_ratio.values[0]
                )
                str_temp += "P-value: {:.2e}<br>".format(
                    curr_line.Pvalue.values[0])
                str_temp += "<extra></extra>"
                cs_data[j][i] = str_temp
    return cs_data


print("I am here 1")
df_x_group = pd.read_csv(
    Path("/home/saveliy/HHAlign/data/hhr_data_by_x_group_df.csv"), index_col=0
).reset_index(drop=True)

print("I am here 2")
cs_data = _make_cs_data(data, df_x_group)

with open(_DATA_LOC / "nice_names_hover.pckl", "wb") as oF:
    pickle.dump(cs_data, oF)
