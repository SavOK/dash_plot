from pathlib import Path
import json, pickle
from typing import Dict, AnyStr, Tuple

import pandas as pd
import numpy as np

import plotly.graph_objects as go
import plotly.express as px
import plotly.offline as pyo


import flask
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate

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


def read_x_groupdf(
    inFile: Path = _DATA_LOC / "hhr_data_by_x_group_df.csv",
) -> pd.DataFrame:
    return pd.read_csv(inFile, index_col=0).reset_index(drop=True)


def read_main_table(inFile: Path = _DATA_LOC / "hhr_clean.csv") -> pd.DataFrame:
    return pd.read_csv(inFile)


def read_ecod_description_df(
    inFile: Path = _DATA_LOC / "ecod_desc.txt",
) -> pd.DataFrame:
    return pd.read_csv(inFile, sep="|")


def read_heatmap_annotation(
    inFile: Path = _DATA_LOC / "heatmap_annotation.pckl",
) -> np.ndarray:
    with open(inFile, "rb") as oF:
        return pickle.load(oF)


names_dict = read_x_names()
data_df = read_matrix_file()
group_df = read_x_groupdf()
main_df = read_main_table()
ecod_df = read_ecod_description_df()
heatmap_annotation = read_heatmap_annotation(_DATA_LOC / "heatmap_annotation_test.pckl")


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
        try:
            dom1, dom2 = group_df[
                (group_df["x_template"] == Y) & (group_df["x_query"] == X)
            ][["template", "query"]].values[0]
        except IndexError:
            return (None, None)
    else:
        try:
            dom1, dom2 = group_df[
                (group_df["x_template"] == X) & (group_df["x_query"] == Y)
            ][["template", "query"]].values[0]
        except IndexError:
            return (None, None)
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
    """get domain id based on X, Y in heatmap

    Arguments:
        X {AnyStr} -- X coord (X group id 1)
        Y {AnyStr} -- Y coord (X group id 2)

    Keyword Arguments:
        ecod_df {pd.DataFrame} -- dataframe with ecod desctiption (default: {ecod_df})
        group_df {pd.DataFrame} -- group data frame (default: {group_df})

    Returns:
        Dict -- dictionary with info
    """
    X = int(X)
    Y = int(Y)
    dom1, dom2 = get_proper_domains_id(X, Y)
    if dom1 is None:
        return None
    info_dict = {"X": X, "Y": Y}
    info_dict.update({"domain1": dom1, "domain2": dom2, "swapFlag": (X > Y)})
    return info_dict


def plot_data_histogram_details(
    info_dict: Dict, main_df: pd.DataFrame = main_df
) -> Tuple:
    """ Generates data for histograms details
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

    # domain1 in group 1
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
    # domain 2 in group 2
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
        # domain 1 in group 2 Swap
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
        # domain 2 in group 1 Swap
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
    # plot data frames
    df_plot1 = pd.DataFrame(
        {
            "prob": dom1_inside_df["prob"].values,
            "type": [f"In Group: ({info_dict['X']}) {x_group_label(info_dict['X'])}"]
            * len(dom1_inside_df),
            "domain": list(_domain_name_gen(dom1_inside_df, info_dict["domain1"])),
            "in_flag": [True] * len(dom1_inside_df),
        }
    )
    df_plot1 = df_plot1.append(
        pd.DataFrame(
            {
                "prob": dom1_outside_df["prob"].values,
                "type": [
                    f"Out Group: ({info_dict['Y']}) {x_group_label(info_dict['Y'])}"
                ]
                * len(dom1_outside_df),
                "domain": _domain_name_gen(dom1_outside_df, info_dict["domain1"]),
                "in_flag": [False] * len(dom1_outside_df),
            }
        )
    ).reset_index(drop=True)
    # plot data frames 2
    df_plot2 = pd.DataFrame(
        {
            "prob": dom2_inside_df["prob"].values,
            "type": [f"In Group: ({info_dict['Y']}) {x_group_label(info_dict['Y'])}"]
            * len(dom2_inside_df),
            "domain": list(_domain_name_gen(dom2_inside_df, info_dict["domain2"])),
            "in_flag": [True] * len(dom2_inside_df),
        }
    )
    df_plot2 = df_plot2.append(
        pd.DataFrame(
            {
                "prob": dom2_outside_df["prob"].values,
                "type": [
                    f"Out Group: ({info_dict['X']}) {x_group_label(info_dict['X'])}"
                ]
                * len(dom2_outside_df),
                "domain": _domain_name_gen(dom2_outside_df, info_dict["domain2"]),
                "in_flag": [False] * len(dom2_outside_df),
            }
        )
    ).reset_index(drop=True)
    return df_plot1, df_plot2


def plot_histogram_details(data, domain_id):
    color_discrete_map = {
        x: c for x, c in zip(sorted(data["type"].unique()), ("#209bf4", "#f47920"))
    }
    fig = px.histogram(
        data,
        x="prob",
        color="type",
        marginal="rug",
        hover_data=["domain"],
        barmode="overlay",
        opacity=0.75,
        nbins=5,
        range_x=[5, 105],
        color_discrete_map=color_discrete_map,
    )
    for i, N in zip((0, 2), ("In Group", "Out Group")):
        fig.data[i].hovertemplate = (
            "probability=%{x}<br>count=%{y}" + "<extra><b>{}</b></extra>".format(N)
        )
    for i in (1, 3):
        fig.data[i].hovertemplate = (
            "<b>Domain ID:</b> %{customdata[0]}<br>"
            + "<b>Probability:</b> %{x}%<br><extra></extra>"
        )
    fig.update_layout(
        dict(
            bargap=0.01,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=-0.3,
                xanchor="center",
                x=0.5,
                title={"text": None},
            ),
            xaxis=dict(title="Probability"),
            yaxis=dict(title="Counts"),
            hovermode="closest",
            title=dict(
                text=f"Domain {domain_id}",
                y=0.94,
                x=0.5,
                xanchor="center",
                yanchor="top",
            ),
        )
    )
    return fig


def plot_histogram(info_dict, FlagFirst: bool = True):
    plot_1_df, plot_2_df = plot_data_histogram_details(info_dict)
    if FlagFirst:
        fig = plot_histogram_details(plot_1_df, info_dict["domain1"])
    else:
        fig = plot_histogram_details(plot_2_df, info_dict["domain2"])
    return fig


heatmap_fig = go.Figure(
    data=go.Heatmap(
        z=data_df.iloc[:20, :20],
        x=data_df.columns[:20],
        y=data_df.index[:20],
        colorscale="Viridis",
        customdata=heatmap_annotation,
        colorbar=dict(
            title=dict(text="HH Probability", font=dict(size=20, family="Arial"),),
        ),
        hovertemplate="%{customdata[2]}<extra></extra>",
    ),
    layout=go.Layout(
        autosize=True,
        hovermode="closest",
        xaxis=dict(type="category", visible=False, title=dict(text="X group")),
        yaxis=dict(type="category", visible=False, title=dict(text="X group")),
    ),
)


def get_group_data(X: AnyStr, Y: AnyStr, main_df: pd.DataFrame = main_df) -> Tuple:
    X = int(X)
    Y = int(Y)
    if X == Y:
        return (
            main_df[(main_df["x_template"] == X) & (main_df["x_query"] == X)],
            None,
            None,
        )
    df1 = main_df[(main_df["x_template"] == X) & (main_df["x_query"] == X)]
    df2 = main_df[(main_df["x_template"] == Y) & (main_df["x_query"] == Y)]
    if X > Y:
        df3 = main_df[(main_df["x_template"] == Y) & (main_df["x_query"] == X)]
    else:
        df3 = main_df[(main_df["x_template"] == X) & (main_df["x_query"] == Y)]
    return (df1, df2, df3)


def get_plotting_data_group(X, Y):
    df1, df2, df3 = get_group_data(X, Y)
    if df1 is None:
        return None
    if not df2 is None:
        df_plot1 = pd.DataFrame(
            {
                "prob": df1["prob"].values,
                "align": df1["Cols"].values,
                "alignr": df1["align_ratio"].values,
                "type": [f"In Group: {X}"] * len(df1),
                "domain1": df1["template"].values,
                "domain2": df1["query"].values,
                "in_flag": [True] * len(df1),
            }
        )
        df_plot1 = df_plot1.append(
            pd.DataFrame(
                {
                    "prob": df2["prob"].values,
                    "align": df2["Cols"].values,
                    "alignr": df2["align_ratio"].values,
                    "type": [f"In Group: {Y}"] * len(df2),
                    "domain1": df2["template"].values,
                    "domain2": df2["query"].values,
                    "in_flag": [True] * len(df2),
                }
            )
        ).reset_index(drop=True)
        df_plot1 = df_plot1.append(
            pd.DataFrame(
                {
                    "prob": df3["prob"].values,
                    "align": df3["Cols"].values,
                    "alignr": df3["align_ratio"].values,
                    "type": [f"Out Group: {X} {Y}"] * len(df3),
                    "domain1": df3["template"].values,
                    "domain2": df3["query"].values,
                    "in_flag": [False] * len(df3),
                }
            )
        ).reset_index(drop=True)
    else:
        df_plot1 = pd.DataFrame(
            {
                "prob": df1["prob"].values,
                "align": df1["Cols"].values,
                "alignr": df1["align_ratio"].values,
                "type": [f"In Group: {X}"] * len(df1),
                "domain1": df1["template"].values,
                "domain2": df1["query"].values,
                "in_flag": [True] * len(df1),
            }
        ).reset_index(drop=True)
    return df_plot1


def plot_histogram_group(df_plot, info_dict: dict):
    if len(df_plot["type"].unique()) == 1:
        color_discrete_map = {
            x: c for x, c in zip(sorted(df_plot["type"].unique()), ["#1f77b4"])
        }
    else:
        color_discrete_map = {
            x: c
            for x, c in zip(
                sorted(df_plot["type"].unique()), ("#1f77b4", "#ff7f0f", "#2ca02c")
            )
        }

    fig = px.histogram(
        df_plot,
        x="prob",
        marginal="rug",
        color="type",
        hover_data=["domain1", "domain2", "align", "alignr"],
        barmode="overlay",
        histnorm="percent",
        opacity=0.60,
        nbins=5,
        range_x=[5, 105],
        color_discrete_map=color_discrete_map,
    )
    if len(df_plot["type"].unique()) == 1:
        fig.data[0].hovertemplate = (
            "Probability: %{x}<br>Percent: %{y:.1f}%"
            + "<extra><b>In Group {}</b></extra>".format(info_dict["X"])
        )
        fig.data[1].hovertemplate = (
            "<b>Domain ID 1:</b> %{customdata[0]}<br>"
            + "<b>Domain ID 2:</b> %{customdata[1]}<br>"
            + "<b>Probability:</b> %{x:2f}<br>"
            + "<b>Align cols:</b> %{customdata[2]}<br>"
            + "<b>Align ratio:</b> %{customdata[3]:.2f}<extra></extra>"
        )
        fig.update_layout(
            title=dict(
                text=f"X Group: ({info_dict['X']}) {x_group_label(info_dict['X'], 40)}",
                y=0.94,
                x=0.5,
                xanchor="center",
                yanchor="top",
            ),
        )
    else:
        for i, N in zip(
            (0, 2, 4),
            (
                "In Group {}".format(info_dict["X"]),
                "In Group {}".format(info_dict["Y"]),
                "Between Groups {} {}".format(info_dict["X"], info_dict["Y"]),
            ),
        ):
            fig.data[i].hovertemplate = (
                "Probability: %{x}<br>Percent: %{y:.1f}%"
                + "<extra><b>{}</b></extra>".format(N)
            )
        for i in (1, 3, 5):
            fig.data[i].hovertemplate = (
                "<b>Domain ID 1:</b> %{customdata[0]}<br>"
                + "<b>Domain ID 2:</b> %{customdata[1]}<br>"
                + "<b>Probability:</b> %{x:2f}<br>"
                + "<b>Align cols:</b> %{customdata[2]}<br>"
                + "<b>Align ratio:</b> %{customdata[3]:.2f}<extra></extra>"
            )
        fig.update_layout(
            {
                "title": dict(
                    text=f"X Group: ({info_dict['X']}) {x_group_label(info_dict['X'], 40)} <br>"
                    + f"X Group: ({info_dict['Y']}) {x_group_label(info_dict['Y'], 40)}",
                    y=0.97,
                    x=0.5,
                    xanchor="center",
                    yanchor="top",
                )
            }
        )

    fig.update_layout(
        dict(
            bargap=0.01,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=-0.3,
                xanchor="center",
                x=0.5,
                title={"text": None},
            ),
            xaxis=dict(title={"text": "Probability"}),
            yaxis=dict(title={"text": "Percent"}),
            hovermode="closest",
        )
    )
    return fig


external_stylesheets = ["https://codepen.io/chriddyp/pen/bWLwgP.css"]
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

app.layout = html.Div(
    id="main-page",
    children=[
        html.H1("Maximum HH Probability", style={"textAlign": "center"}),
        html.Div(
            children=[
                dcc.Graph(
                    id="heat-map",
                    figure=heatmap_fig,
                    config={
                        "displaylogo": False,
                        "modeBarButtonsToRemove": [
                            "lasso2d",
                            "hoverCompareCartesian",
                            "hoverClosestCartesian",
                        ],
                    },
                ),
            ],
        ),
        html.Div(id="details-hist"),
        html.Div(id="groups-hist"),
    ],
)


@app.callback(
    Output("details-hist", "children"),
    [Input("heat-map", "clickData")],
    [State("details-hist", "children")],
)
def details_dev_output(clickData, old_output):
    if clickData is None:
        raise PreventUpdate
    if int(clickData["points"][0]["z"]) <= 10:
        raise PreventUpdate
    X = int(clickData["points"][0]["x"])
    Y = int(clickData["points"][0]["y"])
    coord_dict = {"X": X, "Y": Y}
    info_dict = get_info_dict(**coord_dict)
    if info_dict is None:
        raise PreventUpdate
    plot_1_df, plot_2_df = plot_data_histogram_details(info_dict)
    fig1 = plot_histogram_details(plot_1_df, info_dict["domain1"])
    fig2 = plot_histogram_details(plot_2_df, info_dict["domain2"])
    header_str1 = "**Domain:** [{0}](http://prodata.swmed.edu/ecod/complete/domain/{0}). **X Group:** {1}".format(
        info_dict["domain1"], x_group_label(info_dict["X"], 40)
    )
    header_str2 = "**Domain:** [{0}](http://prodata.swmed.edu/ecod/complete/domain/{0}). **X Group:** {1}".format(
        info_dict["domain2"], x_group_label(info_dict["Y"], 40)
    )
    new_div = html.Div(
        children=[
            html.Div(
                id="details-header",
                children=[dcc.Markdown(header_str1), dcc.Markdown(header_str2)],
                style={"textAlign": "center"},
            ),
            html.Div(
                id="left-panel",
                children=[
                    dcc.Graph(
                        id="figure1", figure=fig1, config={"displayModeBar": False},
                    ),
                ],
                style={"width": "50%", "display": "inline-block"},
            ),
            html.Div(
                id="right-panel",
                children=[
                    dcc.Graph(
                        id="figure2", figure=fig2, config={"displayModeBar": False},
                    )
                ],
                style={"width": "50%", "display": "inline-block"},
            ),
        ],
        style={"margin": "auto"},
    )
    return new_div


@app.callback(
    Output("groups-hist", "children"),
    [Input("heat-map", "clickData")],
    [State("groups-hist", "children")],
)
def group_dev_output(clickData, old_output):
    if clickData is None:
        raise PreventUpdate
    X = int(clickData["points"][0]["x"])
    Y = int(clickData["points"][0]["y"])
    coord_dict = {"X": X, "Y": Y}
    info_dict = get_info_dict(**coord_dict)
    if info_dict is None:
        raise PreventUpdate
    df_plot = get_plotting_data_group(info_dict["X"], info_dict["Y"])
    if df_plot is None:
        raise PreventUpdate
    fig = plot_histogram_group(df_plot, info_dict)
    new_div = html.Div(
        children=[
            html.Div(
                id="group-histogram",
                children=[
                    dcc.Graph(
                        id="gr-hist", figure=fig, config={"displayModeBar": False},
                    ),
                ],
                style={"width": "100%", "display": "inline-block"},
            ),
        ],
        style={"margin": "auto"},
    )
    return new_div


if __name__ == "__main__":
    app.run_server(debug=True)
