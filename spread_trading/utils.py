import hashlib
import os
from enum import Enum
from pathlib import Path
from typing import Callable, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import quandl
from plotly import colors
from plotly.subplots import make_subplots
from scipy import stats

# =============================================================================
# Quandl
# =============================================================================

quandl.ApiConfig.api_key = os.getenv("QUANDL_API_KEY")


def get_security_code(
    security: str,
    feed: str = "OWF",
    expiration: str = "2M",
    time_series: str = "IVM",
    column_index: list = [1, 15, 16],
) -> tuple:
    """Constructs string to retrieve individual security from Quandl.

    Args:
        security: Security in the form of
            `{exchange_code}_{options_code}_{futures_code}`
        feed: Quandl data feed code.
        expiration: Expiration of the contract.
        time_series: Either IVM or IVS.
        column_index: List of the indexes of the columns to be fetched.

    Returns:
        A tuple where first element is a complete string as required by Quandl
        to fetch data and the second element is single key dict containing
        the list of columns to be retrieved.
    """

    query_string = f"{feed}/{security}_{expiration}_{time_series}"

    return (query_string, {"column_index": column_index})


def get_hash(string: str) -> str:
    """Returns md5 hash of string."""

    return hashlib.md5(str(string).encode()).hexdigest()


def fetch_data(query_params: dict, data_dir: str = ".") -> pd.DataFrame:
    """Takes a dict of query parameters and returns a Pandas DataFrame.
    Will return data from disk if it exists or fetch it from Quandl and
    save it to disk if it doesn't exist.

    Args:
        query_params: Dict of query parameters as required by `quandl.get`.
        data_dir: Directory where data files can be stored and retrieved
            from.

    Returns:
        Dataframe as returned from Quandl.
    """

    hash = get_hash(str(query_params))
    filepath = Path(f"{data_dir}/{hash}.csv")

    try:
        data = pd.read_csv(filepath).set_index("Date")
        print(f"Loading {str(filepath)} from disk.")
    except FileNotFoundError:
        print(f"Fetching data from quandl and saving to disk as {str(filepath)}.")
        data = quandl.get(**query_params)
        data.to_csv(filepath)

    return data


# =============================================================================
# Data Preparation and Analysis
# =============================================================================


def parse_column_label(c: str) -> tuple:
    """Parses a column label as returned by Quandl into individual data elements
    to facilitate the flattening of the data.

    Args:
        c: Column label as returned by Quandl.

    Returns:
        Tuple with the following elements:
            feed: Data feed code, i.e., 'OWF'
            sec: Security of the form `{exchange_code}_{options_code}_{futures_code}`
            exp: Contract expiration code, i.e, 'M2019'
            ts: Time series code, i.e., 'IVM"
            series: Label of the data series, i.e., 'Futures`
    """

    elems = c.split("/")
    feed = elems[0]
    descr, series = elems[1].split(" - ")
    exch, op, fu, exp, ts = descr.split("_")
    sec = f"{exch}_{op}_{fu}"

    return tuple([feed, sec, exp, ts, series])


def expand_series(s: pd.Series, date_fmt: str = "%Y-%m-%d") -> pd.DataFrame:
    """Expand original series reuturned from Quandl to include seprate
    series for each data element as returned by `parse_column_label.

    Args:
        s: Original series returned from Quandl
        date_fmt: String formatting for the date index

    Returns
        Dataframe with one column for each data element, including
        the original series.
    """

    labels = ["data_feed", "security", "expiration", "model", "series"]
    values = parse_column_label(s.name)

    df_exp = pd.DataFrame({"date": s.index.values})
    for i, label in enumerate(labels):
        df_exp[label] = values[i]

    df_exp["value"] = s.values
    df_exp["series"] = df_exp["series"].str.lower()

    return df_exp


def get_spread_label(pair: tuple) -> str:
    """Return labels for spread between two securities in the form
    `{exchange code}:{security_2 futures code}-{security_1 futures code}`

    Args:
        pairs: Tuples with two str elements, one for each security
            in the form `{exchange code}_{futures code}_{options_code}`.

    Returns:
        String with label for spread between two securities.
    """

    return f"{pair[0].split('_')[0]}:{pair[1].split('_')[-1]}-{pair[0].split('_')[-1]}"


class ReturnType(Enum):
    LOG: str = "log"
    SIMPLE: str = "simple"
    DIFF: str = "diff"


def get_spread(
    pair: tuple,
    df: pd.DataFrame,
    label_fn: Callable = get_spread_label,
    price_col: str = "adj_future",
    return_type: ReturnType = None,
) -> pd.Series:
    """Calculates spread between two series. The spread will be expressed
    as the second security of the pair less the first one.

    Args:
        pair: Tuple of two str elements, one for each security
            in the form `{exchange code}_{futures code}_{options_code}`.
        df: Pandas dataframe with the price data to be plotted. Assumes series
            are accessible with a tuple of `{(price_col, security)}`.
        label_fn: Function that returns a string label for the new series.
        price_col: Label of column in df by which the prices of the underlying
            securities are accessible.
        return_type: Either `log`, `simple` or `diff` to specify how returns
            are calculated.

    Returns:
        Pandas series of spread.
    """

    label = label_fn(pair)
    price_1, price_2 = (df[(price_col, sec)] for sec in pair)
    spread = (price_2 - price_1).rename(label).dropna()

    if return_type is not None:
        return_type = ReturnType(return_type)

    if return_type == ReturnType.LOG:
        spread = (spread / spread.shift(1)).apply(np.log).dropna()
    elif return_type == ReturnType.SIMPLE:
        spread = spread.pct_change().dropna()
    elif return_type == ReturnType.DIFF:
        spread = spread.diff().dropna()

    return spread


def get_rolling_avg_diffs(
    pairs: tuple,
    df: pd.DataFrame,
    windows: tuple = (30, 90, 180, 360),
    return_type: str = "log",
) -> pd.DataFrame:
    """Calculates differences between a spread and rolling averages of itself over
    provided windows.

    Args:
        pairs: Tuple of two tuples with two str elements, one for each security
            in the form `{exchange code}_{futures code}_{options_code}`.
        df: Pandas dataframe with the price data to be plotted. Assumes series
            are accessible with a tuple of `{(price_col, security)}`.
        windows: Tuple of integers with the windows for the rolling averages.
        return_type: Either `log` or `simple` to specify how returns are calculated.

    Returns:
        Pandas DataFrame of differences
    """

    def get_diffs(pair):
        spread = get_spread(pair, df, return_type=return_type)
        df_diffs = pd.concat(
            [spread - spread.rolling(w).mean() for w in windows], axis=1
        )
        tuples = [(spread.name, w) for w in windows]
        df_diffs.columns = pd.MultiIndex.from_tuples(tuples, names=("spread", "window"))
        return df_diffs

    return pd.concat([get_diffs(pair) for pair in pairs], axis=1)


def get_rolling_kurts(
    pairs: tuple,
    df: pd.DataFrame,
    windows: tuple = (30, 90, 180, 360),
    return_type: str = "log",
) -> pd.DataFrame:
    """Calculates rolling kurtosis over windows.

    Args:
        pairs: Tuple of two tuples with two str elements, one for each security
            in the form `{exchange code}_{futures code}_{options_code}`.
        df: Pandas dataframe with the price data to be plotted. Assumes series
            are accessible with a tuple of `{(price_col, security)}`.
        windows: Tuple of integers with the windows for the rolling kurtosis.
        return_type: Either `log` or `simple` to specify how returns are calculated.

    Returns:
        Pandas DataFrame of differences
    """

    def get_kurts(pair):
        spread = get_spread(pair, df, return_type=return_type)
        df_kurts = pd.concat([spread.rolling(w).kurt() for w in windows], axis=1)
        tuples = [(spread.name, w) for w in windows]
        df_kurts.columns = pd.MultiIndex.from_tuples(tuples, names=("spread", "window"))
        return df_kurts

    return pd.concat([get_kurts(pair) for pair in pairs], axis=1)


# =============================================================================
# Charts
# =============================================================================

COLORS = colors.qualitative.D3


def make_spread_charts(
    pairs: tuple,
    df: pd.DataFrame,
    title_text: str,
    price_col: str = "adj_future",
    date_fmt: str = "%Y-%m-%d",
    fig_size: dict = dict(width=1000, height=500),
) -> go.Figure:
    """Returns figure for side-by-side charts of the prices of the underlying
    securities and spreads. The spread will be expressed as the second security
    of the pair less the first one with the second security appearing in top row.

    Args:
        pairs: Tuple of two tuples with two str elements, one for each security
            in the form `{exchange code}_{futures code}_{options_code}`.
        df: Pandas dataframe with the price data to be plotted. Assumes series
            are accessible with a tuple of `{(price_col, security)}`.
        title_text: Text of overall figure.
        price_col: Label of column in df by which the prices of the underlying
            securities are accessible.

    Returns:
        A plotly Figure ready for plotting

    """

    subplot_titles = (pairs[0][1], pairs[1][1], pairs[0][0], pairs[1][0])
    subplot_titles += tuple(get_spread_label(p) for p in pairs)

    fig = make_subplots(rows=3, cols=2, subplot_titles=subplot_titles)

    y_ranges = {}

    for i, pair in enumerate(pairs):
        for j, security in enumerate(pair):
            series = df[(price_col, security)].dropna()
            fig.append_trace(
                go.Scatter(x=series.index, y=series, name=security),
                row=2 - j,
                col=2 - i,
            )

            # Stores the ranges so they can be set consistently for each security
            # in a pair.
            y_ranges[(2 - j, 2 - i)] = (
                series.max() - series.min(),
                series.min(),
                series.max(),
            )

        spread = get_spread(pair, df=df, price_col=price_col)
        fig.append_trace(
            go.Scatter(
                x=spread.index,
                y=spread,
                name=get_spread_label(pair),
            ),
            row=3,
            col=i + 1,
        )

    beg_date = df.index.min().strftime(date_fmt)
    end_date = df.index.max().strftime(date_fmt)
    title_text = f"{title_text}: {beg_date} - {end_date}"

    fig.update_layout(
        template="none", **fig_size, title_text=title_text, showlegend=False
    )
    fig.update_yaxes(tickprefix="$")

    # Sets y ranges of the underlying securities in a pair to be the same
    # to make it easier to see which movements are causing the spread to change.
    for c in range(1, 3):
        delta = y_ranges[(1, c)][0] - y_ranges[(2, c)][0]
        if delta > 0:
            new_range = (
                y_ranges[(2, c)][1] - delta / 2,
                y_ranges[(2, c)][2] + delta / 2,
            )
            fig.update_yaxes(range=new_range, row=2, col=c)
        else:
            new_range = (
                y_ranges[(1, c)][1] + delta / 2,
                y_ranges[(1, c)][2] - delta / 2,
            )
            fig.update_yaxes(range=new_range, row=1, col=c)

    return fig


def get_moments_annotation(
    s: pd.Series, xref: str, yref: str, x: float, y: float, xanchor: str
) -> go.layout.Annotation:
    """Calculates summary statistics for a series and returns and
    Annotation object.
    """

    labels = [
        ("obs", lambda x: f"{x:>16d}"),
        ("min:max", lambda x: f"{x[0]:>0.2f}:{x[1]:>0.2f}"),
        ("mean", lambda x: f"{x:>13.4f}"),
        ("std", lambda x: f"{x:>15.4f}"),
        ("skewness", lambda x: f"{x:>11.4f}"),
        ("kurtosis", lambda x: f"{x:>13.4f}"),
    ]

    moments = list(stats.describe(s.to_numpy()))
    moments[3] = np.sqrt(moments[3])

    return go.layout.Annotation(
        text=("<br>").join(
            [f"{k[0]:<10}{k[1](moments[i])}" for i, k in enumerate(labels)]
        ),
        align="left",
        showarrow=False,
        xref=xref,
        yref=yref,
        x=x,
        y=y,
        bordercolor="black",
        borderwidth=1,
        borderpad=2,
        bgcolor="white",
        font=dict(size=10),
        xanchor=xanchor,
        yanchor="top",
    )


def get_date_slice_text(
    date_slice: slice, df: pd.DataFrame, date_fmt: str = "%Y-%m-%d"
) -> str:
    """Gets text of date slice.

    Args:
        date_slice: Slice in a form that can be used to index a
            pandas DatetimeIndex.
        df: Dataframe to which date_slice will be applied. Used to get
            min or max dates if slice is open ended.
        date_fmt: String formatting to be applied to dates retrieved
            from df.

    Returns:
        String in the form of `{start} - {end}` where start and end
            are in the form specified in date_fmt

    """

    if date_slice.start is None:
        start = df.index.min().strftime(date_fmt)
    else:
        start = date_slice.start

    if date_slice.stop is None:
        stop = df.index.max().strftime(date_fmt)
    else:
        stop = date_slice.stop

    return f"{start} - {stop}"


def make_tail_charts(
    pairs: tuple,
    df: pd.DataFrame,
    title_text: str,
    date_slices: Tuple[slice, slice] = None,
    price_col: str = "adj_future",
    date_fmt: str = "%Y-%m-%d",
    fig_size: dict = dict(width=1000, height=500),
    return_type: str = "log",
    moments_xanchors: Tuple[str, str] = ("right", "right"),
) -> go.Figure:
    """Returns figure for side-by-side scatter plots of daily returns overlayed on
    the normal distribution with kurtosis statistics.

    Args:
        pairs: Tuple of two tuples with two str elements, one for each security
            in the pair of the form `{exchange code}_{futures code}_{options_code}`.
        df: Pandas dataframe with the price data to be plotted. Assumes series
            are accessible with a tuple of `{(price_col, security)}`.
        title_text: Text of overall figure.
        date_slices: Date ranges for charts, provided in the form of a tuple of slices
            that can be used to index a Pandas DatetimeIndex. Date range will be
            removed from the figure title and subplot date ranges added to subplot
            titles.
        price_col: Label of column in df by which the prices of the underlying
            securities are accessible.
        return_type: Either `log`, `simple` or `diff` to specify how returns
            are calculated.
        moments_xanchors: Tuple of strings of `left` or `right` indicating which side
            of the charts to display the moments annotation on.

    Returns:
        A plotly Figure ready for plotting

    """

    subplot_titles = tuple(get_spread_label(p) for p in pairs)

    fig = make_subplots(rows=1, cols=2, subplot_titles=subplot_titles)

    for i, pair in enumerate(pairs):

        if date_slices is not None:
            date_slice = date_slices[i]

            ds_text = get_date_slice_text(date_slice, df)
            text = f"{subplot_titles[i]}: {ds_text}"
            fig.layout.annotations[i].update(text=text)
        else:
            date_slice = slice(None, None)

        if pairs[0] == pairs[1]:
            line_color = COLORS[0]
            normal_line_color = COLORS[1]
        else:
            line_color = COLORS[i * 2]
            normal_line_color = COLORS[i * 2 + 1]

        spread = get_spread(
            pair, df=df.loc[date_slice], price_col=price_col, return_type=return_type
        )
        returns = pd.cut(spread, 100).value_counts().sort_index()
        midpoints = returns.index.map(lambda interval: interval.right).to_numpy()
        norm_dist = stats.norm.pdf(midpoints, loc=spread.mean(), scale=spread.std())

        fig.append_trace(
            go.Scatter(
                x=[interval.mid for interval in returns.index],
                y=returns / returns.sum() * 100,
                name=spread.name,
                line_color=line_color,
            ),
            row=1,
            col=i + 1,
        )

        fig.append_trace(
            go.Scatter(
                x=[interval.mid for interval in returns.index],
                y=norm_dist / norm_dist.sum() * 100,
                name="Normal Distribution",
                line_color=normal_line_color,
            ),
            row=1,
            col=i + 1,
        )

        x = midpoints.max() - 0.05 * (midpoints.max() - midpoints.min())
        if moments_xanchors[i] == "left":
            x = midpoints.min() + 0.05 * (midpoints.max() - midpoints.min())
        y = (returns / returns.sum() * 100).max()
        fig.add_annotation(
            get_moments_annotation(
                spread,
                xref=f"x{i+1}",
                yref=f"y{i+1}",
                x=x,
                y=y,
                xanchor=moments_xanchors[i],
            )
        )

    if date_slice == slice(None, None):
        title_text = (
            f"{title_text}: {df.index.min().strftime(date_fmt)}"
            f" - {df.index.max().strftime(date_fmt)}"
        )

    fig.update_layout(
        template="none",
        **fig_size,
        title_text=title_text,
        showlegend=False,
    )
    fig.update_yaxes(tickprefix="%")

    for d in fig["data"]:
        xaxis = d["xaxis"]
        xaxis = f"xaxis{xaxis.replace('x', '')}"
        tickvals = d["x"][::5]
        fig.update_layout(
            {xaxis: {"tickmode": "array", "tickvals": tickvals, "tickformat": ".4f"}}
        )

    return fig


def make_qq_charts(
    pairs: tuple,
    df: pd.DataFrame,
    title_text: str,
    date_slices: Tuple[slice, slice] = None,
    price_col: str = "adj_future",
    date_fmt: str = "%Y-%m-%d",
    fig_size: dict = dict(width=1000, height=500),
    return_type: str = "log",
) -> go.Figure:
    """Returns figure for side-by-side q-q plots of daily returns.

    Args:
        pairs: Tuple of two tuples with two str elements, one for each security
            in the pair of the form `{exchange code}_{futures code}_{options_code}`.
        df: Pandas dataframe with the price data to be plotted. Assumes series
            are accessible with a tuple of `{(price_col, security)}`.
        title_text: Text of overall figure.
        date_slices: Date ranges for charts, provided in the form of a tuple of slices
            that can be used to index a Pandas DatetimeIndex. Date range will be
            removed from the figure title and subplot date ranges added to subplot
            titles.
        price_col: Label of column in df by which the prices of the underlying
            securities are accessible.
        return_type: Either `log`, `simple` or `diff` to specify how returns
            are calculated.

    Returns:
        A plotly Figure ready for plotting

    """

    subplot_titles = tuple(get_spread_label(p) for p in pairs)

    fig = make_subplots(rows=1, cols=2, subplot_titles=subplot_titles)

    for i, pair in enumerate(pairs):

        if date_slices is not None:
            date_slice = date_slices[i]

            ds_text = get_date_slice_text(date_slice, df)
            text = f"{subplot_titles[i]}: {ds_text}"
            fig.layout.annotations[i].update(text=text)
        else:
            date_slice = slice(None, None)

        if pairs[0] == pairs[1]:
            line_color = COLORS[0]
        else:
            line_color = COLORS[i]

        returns = get_spread(
            pair, df=df.loc[date_slice], price_col=price_col, return_type=return_type
        )

        returns_norm = ((returns - returns.mean()) / returns.std()).sort_values()
        norm_dist = pd.Series(
            list(map(stats.norm.ppf, np.linspace(0.001, 0.999, len(returns)))),
            name="normal",
        )

        fig.append_trace(
            go.Scatter(
                x=norm_dist,
                y=returns_norm,
                name=returns.name,
                mode="markers",
                marker=dict(color=line_color),
            ),
            row=1,
            col=i + 1,
        )

        x = norm_dist.min() + 0.3 * (norm_dist.max() - norm_dist.min())
        y = returns_norm.max()
        fig.add_annotation(
            get_moments_annotation(
                returns, xref=f"x{i+1}", yref=f"y{i+1}", x=x, y=y, xanchor="right"
            )
        )

    if date_slice == slice(None, None):
        title_text = (
            f"{title_text}: {df.index.min().strftime(date_fmt)}"
            f" - {df.index.max().strftime(date_fmt)}"
        )

    updates = dict(
        template="none",
        **fig_size,
        title_text=title_text,
        showlegend=False,
    )
    for i in range(1, 3):
        updates[f"xaxis{i}"] = dict(title="normal")
        updates[f"yaxis{i}"] = dict(title="returns")

    fig.update_layout(**updates)

    for d in fig["data"]:
        xaxis = d["xaxis"]
        xaxis = f"xaxis{xaxis.replace('x', '')}"
        tickvals = np.linspace(-3, 3, 7)
        fig.update_layout(
            {xaxis: {"tickmode": "array", "tickvals": tickvals, "tickformat": ".0f"}}
        )

    return fig


def make_rolling_charts(
    df: pd.DataFrame,
    title_text: str,
    date_fmt: str = "%Y-%m-%d",
    fig_size: dict = dict(width=1000, height=1000),
) -> go.Figure:
    """Returns figure for side-by-side charts of the difference between the spreads and
    . The spread will be expressed as the second security
    of the pair less the first one with the second security appearing in top row.

    Args:
        df: Pandas dataframe with the price data to be plotted. Assumes series
            are accessible with a tuple of `{(price_col, security)}`.
        title_text: Text of overall figure.

    Returns:
        A plotly Figure ready for plotting

    """

    cols, rows = df.columns.levels
    subplot_titles = [[f"{col} - {row:02d} days" for row in rows] for col in cols]
    subplot_titles = sum([list(z) for z in zip(*subplot_titles)], [])

    fig = make_subplots(
        rows=len(rows),
        cols=len(cols),
        subplot_titles=subplot_titles,
        shared_xaxes=True,
    )

    counter = 0
    for i, col in enumerate(cols):
        for j, row in enumerate(rows):
            counter += 1
            series = df[(col, row)].dropna()
            fig.append_trace(
                go.Scatter(
                    x=series.index,
                    y=series,
                    name=f"{row:03d} days",
                    line=dict(color=COLORS[counter - 1]),
                ),
                row=j + 1,
                col=i + 1,
            )

            fig.append_trace(
                go.Scatter(
                    x=series.index,
                    y=[series.mean()] * len(series),
                    name="mean",
                    line=dict(color="black", width=1),
                ),
                row=j + 1,
                col=i + 1,
            )

    updates = dict(
        template="none",
        **fig_size,
        title_text=title_text,
        showlegend=False,
    )
    for i in range(1, len(subplot_titles) + 1):
        updates[f"xaxis{i}"] = dict(showticklabels=True)
        updates[f"yaxis{i}"] = dict(showticklabels=True)

    fig.update_layout(**updates)

    return fig
