from pathlib import Path
import json
from typing import Dict, AnyStr, Tuple

import pandas as pd
import numpy as np

import plotly.graph_objects as go
import plotly.figure_factory as ff
import plotly.express as px

import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State

_OPTIONS = {
    "win_loc": Path("C:\\Users\\Saveliy\\Projects\\HHAlign"),
    "unix_loc": Path("/home/saveliy/HHAlign"),
}

_ENV = "win"
if _ENV == "win":
    _MAIN_LOC = _OPTIONS["win_loc"] / "dash_plot"
else:
    _MAIN_LOC = _OPTIONS["unix_loc"] / "dash_plot"
_DATA_LOC = _MAIN_LOC / "data_plots"


def read_x_names(
    inFile: Path = _DATA_LOC / "x_group_names.json",
) -> Dict[AnyStr, AnyStr]:
    with open(inFile) as oF:
        return json.load(oF)


def read_matrix_file(
    inFile: Path = _DATA_LOC / "probability_matrix.txt",
) -> pd.DataFrame:
    return pd.read_csv(inFile, index_col=0)


def read_x_group_df(
    inFile: Path = _DATA_LOC / "hhr_data_by_x_group_df.csv",
) -> pd.DataFrame:
    return pd.read_csv(inFile, index_col=0).reset_index(drop=True)


def read_main_table(inFile: Path = _DATA_LOC / "hhr_clean.csv") -> pd.DataFrame:
    return pd.read_csv(inFile)


def read_ecod_description_df(
    inFile: Path = _DATA_LOC / "ecod_desc.txt",
) -> pd.DataFrame:
    return pd.read_csv(inFile, sep="|")


names_dict = read_x_names()
plot_df = read_matrix_file()
group_df = read_x_group_df()
main_df = read_main_table()
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


def get_info_dict(
    X: AnyStr,
    Y: AnyStr,
    ecod_df: pd.DataFrame = ecod_df,
    group_df: pd.DataFrame = group_df,
) -> Dict:
    X = int(X)
    Y = int(Y)
    dom1, dom2 = get_proper_domains_id(X, Y)
    info_dict = {"X": X, "Y": Y}
    # step 1 get domain id
    info_dict.update({"domain1": dom1, "domain2": dom2, "swapFlag": (X > Y)})
    return info_dict


def get_plot_data(info_dict: Dict, main_df: pd.DataFrame = main_df) -> Tuple:
    """


    Arguments:
        info_dict {Dict} -- [description]

    Keyword Arguments:
        main_df {pd.DataFrame} -- [description] (default: {main_df})

    Returns:
        Tuple -- [description]
    """
    def _domain_name_gen(short_df: pd.DataFrame, dom1: str):
        for row in short_df[["template", "query"]].itertuples():
            if row[1] != dom1:
                yield row[1]
            else:
                yield row[2]

    dom1_inside_df = main_df[
        (
            (main_df["template"] == info_dict["domain1"])
            | (main_df["query"] == info_dict["domain1"])
        )
        & (
            (main_df["x_template"] == int(info_dict["X"]))
            & (main_df["x_query"] == int(info_dict["X"]))
        )
    ]
    dom2_inside_df = main_df[
        (
            (main_df["template"] == info_dict["domain2"])
            | (main_df["query"] == info_dict["domain2"])
        )
        & (
            (main_df["x_template"] == int(info_dict["Y"]))
            & (main_df["x_query"] == int(info_dict["Y"]))
        )
    ]
    if info_dict["swapFlag"]:
        dom1_outside_df = main_df[
            (
                (main_df["template"] == info_dict["domain1"])
                | (main_df["query"] == info_dict["domain1"])
            )
            & (
                (main_df["x_template"] == int(info_dict["Y"]))
                & (main_df["x_query"] == int(info_dict["X"]))
            )
        ]
        dom2_outside_df = main_df[
            (
                (main_df["template"] == info_dict["domain2"])
                | (main_df["query"] == info_dict["domain2"])
            )
            & (
                (main_df["x_template"] == int(info_dict["Y"]))
                & (main_df["x_query"] == int(info_dict["X"]))
            )
        ]
    else:
        dom1_outside_df = main_df[
            (
                (main_df["template"] == info_dict["domain1"])
                | (main_df["query"] == info_dict["domain1"])
            )
            & (
                (main_df["x_template"] == int(info_dict["X"]))
                & (main_df["x_query"] == int(info_dict["Y"]))
            )
        ]
        dom2_outside_df = main_df[
            (
                (main_df["template"] == info_dict["domain2"])
                | (main_df["query"] == info_dict["domain2"])
            )
            & (
                (main_df["x_template"] == int(info_dict["X"]))
                & (main_df["x_query"] == int(info_dict["Y"]))
            )
        ]
    df_plot1 = pd.DataFrame(
        {
            "prob": dom1_inside_df["prob"].values,
            "mark": [f"In Group: ({info_dict['X']}) {x_group_label(info_dict['X'])}"]
            * len(dom1_inside_df),
            "domain": list(_domain_name_gen(dom1_inside_df, info_dict["domain1"])),
        }
    )
    df_plot1 = df_plot1.append(
        pd.DataFrame(
            {
                "prob": dom1_outside_df["prob"].values,
                "mark": [
                    f"Out Group: ({info_dict['Y']}) {x_group_label(info_dict['Y'])}"
                ]
                * len(dom1_outside_df),
                "domain": _domain_name_gen(dom1_outside_df, info_dict["domain1"]),
            }
        )
    ).reset_index(drop=True)

    df_plot2 = pd.DataFrame(
        {
            "prob": dom2_inside_df["prob"].values,
            "mark": [f"In Group: ({info_dict['X']}) {x_group_label(info_dict['X'])}"]
            * len(dom2_inside_df),
            "domain": list(_domain_name_gen(dom2_inside_df, info_dict["domain2"])),
        }
    )
    df_plot2 = df_plot2.append(
        pd.DataFrame(
            {
                "prob": dom2_outside_df["prob"].values,
                "mark": [
                    f"Out Group: ({info_dict['Y']}) {x_group_label(info_dict['Y'])}"
                ]
                * len(dom2_outside_df),
                "domain": _domain_name_gen(dom2_outside_df, info_dict["domain2"]),
            }
        )
    ).reset_index(drop=True)
    return df_plot1, df_plot2


test_case = {"X": "11", "Y": "52"}
info_dict = get_info_dict(**test_case)
# step 2 make partial df
plot1_1, plot2_1 = get_plot_data(info_dict)
test_case["X"] = "52"
test_case["Y"] = "11"
info_dict = get_info_dict(**test_case)
plot1_2, plot2_2 = get_plot_data(info_dict)
