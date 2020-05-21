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


def read_x_groupdf(
    inFile: Path = _DATA_LOC / "hhr_data_by_x_group_df.csv",
) -> pd.DataFrame:
    return pd.read_csv(inFile, index_col=0).reset_index(drop=True)


def read_main_table(inFile: Path = _DATA_LOC / "hhr_clean.csv") -> pd.DataFrame:
    return pd.read_csv(inFile)


def _nice_lables(
    x: int, cut: int = 20, name_dict: Dict[AnyStr, AnyStr] = None
) -> AnyStr:
    if name_dict is None:
        name_dict = read_x_names()

    name = name_dict[str(x)]
    if len(name) > cut:
        return f"{name[:cut-3]}..."
    else:
        return name


names_dict = read_x_names()
data_df = read_matrix_file()
group_df = read_x_groupdf()
main_df = read_main_table()


def _extract_plot_hist_data(
    domain: str, x_group1: int, x_group2: int, Flag: bool
) -> Tuple[go.Figure]:
    x_gr1 = _nice_lables(x_group1, 20).capitalize()
    x_gr2 = _nice_lables(x_group2, 20).capitalize()
    if not Flag:
        df11 = main_df[
            (main_df["template"] == domain) & (main_df["x_query"] == x_group1)
        ]
        df12 = main_df[
            (main_df["template"] == domain) & (main_df["x_query"] == x_group2)
        ]
        df_plot1 = pd.DataFrame(
            {
                "prob": df11["prob"].values,
                "mark": [f"In Group: ({x_group1}) {x_gr1}"] * len(df11),
                "domain": df11["query"].values,
            }
        )
        df_plot1 = df_plot1.append(
            pd.DataFrame(
                {
                    "prob": df12["prob"].values,
                    "mark": [f"Out Group: ({x_group2}) {x_gr2}"] * len(df12),
                    "domain": df12["query"].values,
                }
            )
        )

        fig = px.histogram(
            df_plot1, x="prob", color="mark", marginal="rug", hover_data=["domain"]
        )
        print(x_gr1, x_gr2, domain)
        print(df_plot1)
        print(fig.data)
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
                legend=dict(
                    orientation="h", yanchor="bottom", y=-0.3, xanchor="center", x=0.5
                ),
                xaxis=dict(title="Probability"),
                yaxis=dict(title="Counts"),
                hovermode="closest",
                title=dict(
                    text=f"Domain {domain}",
                    y=0.94,
                    x=0.5,
                    xanchor="center",
                    yanchor="top",
                ),
            )
        )
    else:
        df21 = main_df[
            (main_df["query"] == domain) & (main_df["x_template"] == x_group2)
        ]
        df22 = main_df[(main_df["query"] == domain) & (main_df["x_query"] == x_group1)]
        df_plot2 = pd.DataFrame(
            {
                "prob": df21["prob"].values,
                "mark": [f"In Group: ({x_group2}) {x_gr2}"] * len(df21),
                "domain": df21["query"].values,
            }
        )
        df_plot2 = df_plot2.append(
            pd.DataFrame(
                {
                    "prob": df22["prob"].values,
                    "mark": [f"Out Group: ({x_group1}) {x_gr2}"] * len(df22),
                    "domain": df22["query"].values,
                }
            )
        )
        fig = px.histogram(
            df_plot2, x="prob", color="mark", marginal="rug", hover_data=["domain"]
        )
        for i, N in zip((0, 2), ("In Group", "Out Group")):
            fig.data[i].hovertemplate = (
                "probability=%{x}<br>count=%{y}" + "<extra><b>{}</b></extra>".format(N)
            )
        fig.data[1].hovertemplate = (
            "<b>Domain ID:</b> %{customdata[0]}<br>"
            + "<b>Probability:</b> %{x}%<br><extra></extra>"
        )
        fig.data[3].hovertemplate = (
            "<b>Domain ID:</b> %{customdata[0]}<br>"
            + "<b>Probability:</b> %{x}%<br><extra></extra>"
        )
        fig.update_layout(
            dict(
                legend=dict(
                    orientation="h", yanchor="bottom", y=-0.3, xanchor="center", x=0.5
                ),
                xaxis=dict(title="Probability"),
                yaxis=dict(title="Counts"),
                hovermode="closest",
                title=dict(
                    text=f"Domain {domain}",
                    y=0.94,
                    x=0.5,
                    xanchor="center",
                    yanchor="top",
                ),
            )
        )
    return fig


cs_part_1 = np.ndarray(shape=(20, 20), dtype=np.object)
cs_part_2 = np.ndarray(shape=(20, 20), dtype=np.object)
cs_part_3 = np.ndarray(shape=(20, 20), dtype=np.object)
for i, I in enumerate(sorted(int(x) for x in data_df.index[:20])):
    I_L = _nice_lables(I, 30).capitalize()
    for j, J in enumerate(sorted(int(x) for x in data_df.columns[:20])):
        J_L = _nice_lables(J, 30).capitalize()
        cs_part_1[i][j] = I
        cs_part_2[i][j] = J
        str_out = "Group: ({}) {} <br>".format(I, I_L)
        str_out += "Group: ({}) {} <br>".format(J, J_L)
        if I > J:
            curr_data = group_df[
                (group_df["x_template"] == J) & (group_df["x_query"] == I)
            ]
            if not curr_data.empty:
                str_out += "Domain : {} <br>".format(curr_data["query"].values[0])
                str_out += "Domain : {} <br>".format(curr_data["template"].values[0])
        else:
            curr_data = group_df[
                (group_df["x_template"] == I) & (group_df["x_query"] == J)
            ]
            if not curr_data.empty:
                str_out += "Domain : {} <br>".format(curr_data["template"].values[0])
                str_out += "Domain : {} <br>".format(curr_data["query"].values[0])
        if not curr_data.empty:
            str_out += "Probability: {:.1f}% <br>".format(curr_data["prob"].values[0])
            str_out += "P-value: {:.1e} <br>".format(curr_data["Pvalue"].values[0])
            str_out += "Align Length: {} <br>".format(int(curr_data["Cols"].values[0]))
            str_out += "Align Ratio: {:.2f} <br>".format(
                curr_data["align_ratio"].values[0]
            )
        cs_part_3[i][j] = str_out


heatmap_fig = go.Figure(
    data=go.Heatmap(
        z=data_df.iloc[:20, :20],
        x=data_df.columns[:20],
        y=data_df.index[:20],
        colorscale="Viridis",
        customdata=np.dstack((cs_part_1, cs_part_2, cs_part_3)),
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

external_stylesheets = ["https://codepen.io/chriddyp/pen/bWLwgP.css"]
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

app.layout = html.Div(
    id="main-page",
    children=[
        html.H1("Maximum HH Probability", style={"textAlign": "center"}),
        html.Div(children=[dcc.Graph(id="heat-map", figure=heatmap_fig,),],),
        html.Div(
            id="details-page",
            children=[
                # html.Pre(id="output"),
                html.Div(id="left-panel", children=[dcc.Graph(id="figure1")]),
                # html.Div(id="right-panel", children=[dcc.Graph(id="figure2")]),
            ],
        ),
    ],
)


# @app.callback(Output("output", "children"), [Input("heat-map", "clickData")])
def print_click_data(clickData):
    if int(clickData["points"][0]["z"]) <= 10:
        return
    if int(clickData["points"][0]["x"]) > int(clickData["points"][0]["y"]):
        flagSwap = True
        X = int(clickData["points"][0]["y"])
        Y = int(clickData["points"][0]["x"])
    else:
        flagSwap = False
        X = int(clickData["points"][0]["x"])
        Y = int(clickData["points"][0]["y"])

    domain1 = group_df[(group_df.x_template == X) & (group_df.x_query == Y)][
        "query"
    ].values[0]
    domain2 = group_df[(group_df.x_template == X) & (group_df.x_query == Y)][
        "template"
    ].values[0]
    clickData.update({"template": domain1, "query": domain2})
    clickData.update({"Fl": flagSwap})
    return json.dumps(clickData, indent=2)


@app.callback(Output("figure1", "figure"), [Input("heat-map", "clickData")])
def figure1_callback(clickData):
    if int(clickData["points"][0]["z"]) <= 10:
        return
    X = int(clickData["points"][0]["x"])
    Y = int(clickData["points"][0]["y"])
    gr1_name = _nice_lables(X, 20).capitalize()
    gr2_name = _nice_lables(Y, 20).capitalize()
    if Y > X:
        domain1 = group_df[(group_df.x_template == X) & (group_df.x_query == Y)][
            "template"
        ].values[0]
        domain2 = group_df[(group_df.x_template == X) & (group_df.x_query == Y)][
            "query"
        ].values[0]
    else:
        domain1 = group_df[(group_df.x_template == Y) & (group_df.x_query == X)][
            "query"
        ].values[0]
        domain2 = group_df[(group_df.x_template == Y) & (group_df.x_query == X)][
            "template"
        ].values[0]

    if Y > X:
        in_group1_df = main_df[
            ((main_df.x_template == X) & (main_df.x_query == X))
            & ((main_df["query"] == domain1) | (main_df["template"] == domain1))
        ]
    else:
        in_group1_df = main_df[
            ((main_df.x_template == Y) & (main_df.x_query == Y))
            & ((main_df["query"] == domain2) | (main_df["template"] == domain2))
        ]
    if Y > X:
        out_group1_df = main_df[
            ((main_df.x_template == X) & (main_df.x_query == Y))
            & ((main_df["query"] == domain1) | (main_df["template"] == domain1))
        ]
    else:
        out_group1_df = main_df[
            ((main_df.x_template == Y) & (main_df.x_query == X))
            & ((main_df["query"] == domain1) | (main_df["template"] == domain1))
        ]

    df_plot = pd.DataFrame(
        {
            "prob": in_group1_df["prob"].values,
            "mark": [f"In Group: ({X}) {gr1_name}"] * len(in_group1_df),
            "domain": in_group1_df["query"].values,
        }
    )
    df_plot = df_plot.append(
        pd.DataFrame(
            {
                "prob": out_group1_df["prob"].values,
                "mark": [f"Out Group: ({Y}) {gr2_name}"] * len(out_group1_df),
                "domain": out_group1_df["query"].values,
            }
        )
    )

    fig = px.histogram(
        df_plot, x="prob", color="mark", marginal="rug", hover_data=["domain"]
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
            legend=dict(
                orientation="h", yanchor="bottom", y=-0.3, xanchor="center", x=0.5
            ),
            xaxis=dict(title="Probability"),
            yaxis=dict(title="Counts"),
            hovermode="closest",
            title=dict(
                text=f"Domain {domain1}",
                y=0.94,
                x=0.5,
                xanchor="center",
                yanchor="top",
            ),
        )
    )

    # out_group1_df = main_df[
    #     (main_df["query"] == domain1) & (main_df["x_template"] == X)
    # ]
    # if in_group1_df.empty:
    #     in_group1_df = main_df[
    #         (main_df["template"] == domain1) & (main_df["x_query"] == X)
    #     ]
    # in_group2_df = main_df[(main_df["template"] == domain2) & (main_df["x_query"] == Y)]
    # out_group2_df = main_df[
    #     (main_df["query"] == domain2) & (main_df["x_template"] == X)
    # ]
    # if in_group2_df.empty:
    #     in_group2_df = main_df[
    #         (main_df["template"] == domain2) & (main_df["x_query"] == X)
    #     ]

    print("Data:", X, Y, domain1, domain2)
    print("Group Names:", gr1_name, gr2_name)

    # print(in_group1_df.head(), out_group1_df.head())
    # print("GROUP 2")
    # print(in_group2_df.head(), out_group2_df.head())

    # fig = _extract_plot_hist_data(domain1, X, Y, False)
    return fig


# main_df[(main_df["template"] == l["domain1"]) & (main_df['x_query']==l['Y'])].head()


def plot_histogram_left(x):
    pass


if __name__ == "__main__":
    app.run_server(debug=True)
