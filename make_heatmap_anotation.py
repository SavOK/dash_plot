"""
Generates hover info for main heatmap
"""
from pathlib import Path
import json
import pickle
from typing import List, Dict, AnyStr, Tuple

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


def read_matrix_file(
    inFile: Path = _DATA_LOC / "probability_matrix.txt",
) -> pd.DataFrame:
    return pd.read_csv(inFile, index_col=0)


def read_x_groupdf(
    inFile: Path = _DATA_LOC / "hhr_data_by_x_group_df.csv",
) -> pd.DataFrame:
    return pd.read_csv(inFile, index_col=0).reset_index(drop=True)


def read_ecod_description_df(
    inFile: Path = _DATA_LOC / "ecod_desc.txt",
) -> pd.DataFrame:
    return pd.read_csv(inFile, sep="|")


names_dict = read_x_names()
data_df = read_matrix_file()
group_df = read_x_groupdf()
ecod_df = read_ecod_description_df()


def x_group_label(
    x_gr: int, cut: int = 20, name_dict: Dict[AnyStr, AnyStr] = names_dict
) -> AnyStr:
    """Convet X group in nice printeble format
    Arguments:
        x_gr {int} -- X group id

    Keyword Arguments:
        cut {int} -- length of the output string (default: {20})
        name_dict {Dict[AnyStr, AnyStr]} -- dictionary X group id -> X group name] (default: {names_dict})

    Returns:
        AnyStr -- name of X group
    """
    name = name_dict[str(x_gr)]
    if len(name) > cut:
        return f"{name[:cut-3]}..."
    else:
        return name


def get_proper_domains_id(
    x_gr1: int,
    x_gr2: int,
    ecod_df: pd.DataFrame = ecod_df,
    group_df: pd.DataFrame = group_df,
) -> Tuple[AnyStr, AnyStr]:
    """get domain id based on x group

    Arguments:
        x_gr1 {int} -- X group id 1
        x_gr2 {int} -- X group id 2

    Keyword Arguments:
        ecod_df {pd.DataFrame} -- df ecod description (default: {ecod_df})
        group_df {pd.DataFrame} -- group df (default: {group_df})

    Returns:
        Tuple[AnyStr, AnyStr] -- domainID1, domainID2
    """
    X = int(x_gr1)
    Y = int(x_gr2)
    if X > int(x_gr2):
        dom1, dom2 = group_df[
            (group_df["x_template"] == Y) & (group_df["x_query"] == X)
        ][["template", "query"]].values[0]
    else:
        dom1, dom2 = group_df[
            (group_df["x_template"] == X) & (group_df["x_query"] == Y)
        ][["template", "query"]].values[0]
    dom1_x = ecod_df[ecod_df["ecod_domain_id"] == dom1]["f_id"].values[0].split(".")[0]
    dom2_x = ecod_df[ecod_df["ecod_domain_id"] == dom2]["f_id"].values[0].split(".")[0]
    corr_dict = {int(dom1_x): dom1, int(dom2_x): dom2}
    return corr_dict[X], corr_dict[Y]


def heatmap_annotation(
    df_plot, df_x_group, x_range: int = None, y_range: int = None
) -> List[List]:
    if x_range is None:
        x_range = len(df_plot.columns)
    if y_range is None:
        y_range = len(df_plot.index)

    cs_part_1 = np.ndarray(shape=(y_range, x_range), dtype=np.object)
    cs_part_2 = np.ndarray(shape=(y_range, x_range), dtype=np.object)
    cs_part_3 = np.ndarray(shape=(y_range, x_range), dtype=np.object)
    for i, I in enumerate(sorted(int(x) for x in df_plot.index[:y_range])):
        I_L = x_group_label(I, 30).capitalize()
        for j, J in enumerate(sorted(int(x) for x in df_plot.columns[:x_range])):
            J_L = x_group_label(J, 30).capitalize()
            cs_part_1[i][j] = I
            cs_part_2[i][j] = J
            str_out = "Group: ({}) {} <br>".format(I, I_L)
            str_out += "Group: ({}) {} <br>".format(J, J_L)
            if I > J:
                curr_data = df_x_group[
                    (df_x_group["x_template"] == J) & (df_x_group["x_query"] == I)
                ]
            else:
                curr_data = df_x_group[
                    (df_x_group["x_template"] == I) & (df_x_group["x_query"] == J)
                ]
            if curr_data.empty:
                cs_part_3[i][j] = str_out
                continue
            dom1, dom2 = get_proper_domains_id(I, J)
            str_out += "Domain : {} <br>".format(dom1)
            str_out += "Domain : {} <br>".format(dom2)
            str_out += "Probability: {:.1f}% <br>".format(curr_data["prob"].values[0])
            str_out += "P-value: {:.1e} <br>".format(curr_data["Pvalue"].values[0])
            str_out += "Align Length: {} <br>".format(int(curr_data["Cols"].values[0]))
            str_out += "Align Ratio: {:.2f} <br>".format(
                curr_data["align_ratio"].values[0]
            )
            cs_part_3[i][j] = str_out
    return np.dstack((cs_part_1, cs_part_2, cs_part_3))


cs_data = heatmap_annotation(data_df, group_df)

with open(_DATA_LOC / "heatmap_annotation.pckl", "wb") as oF:
    pickle.dump(cs_data, oF)
