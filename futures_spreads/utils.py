import hashlib
import os
from pathlib import Path
from typing import Callable

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import quandl
from scipy.stats import norm

quandl.ApiConfig.api_key = os.getenv("QUANDL_API_KEY")


# =============================================================================
# Quandl
# =============================================================================


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
        time_series: Eitehr IVM or IVS.
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


def fetch_data(
    query_params: dict, data_dir: str = "futures_spreads/data"
) -> pd.DataFrame:
    """Takes a dict of query parameters and returns a Pandas DataFrame.
    Will return data from disk if it exists or fetch it from Quandl and
    save it to disk if it doesn't exist.

    Args:
        query_params: Dict of query parameters as required by `quandl.get`.
        data_dir: Directory where data files can be stored and retrieved
            from.

    Returns:
        Dataframe as returned from Quandle.
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
# Data Preparation
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


def get_spread(
    pair: tuple,
    df: pd.DataFrame,
    label_fn: Callable = get_spread_label,
    price_col: str = "adj_future",
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


    Returns:
        Pandas series of spreadd.
    """

    label = label_fn(pair)
    price_1, price_2 = (df[(price_col, sec)] for sec in pair)

    return (price_2 - price_1).rename(label)


# =============================================================================
# Charts
# =============================================================================


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
            series = df[(price_col, security)]
            fig.append_trace(
                go.Scatter(x=df.index, y=series, name=security), row=2 - j, col=2 - i
            )

            # Stores the ranges so they can be set consistently for each security.
            y_ranges[(2 - j, 2 - i)] = (
                series.max() - series.min(),
                series.min(),
                series.max(),
            )

        fig.append_trace(
            go.Scatter(
                x=df.index,
                y=get_spread(pair, df=df, price_col=price_col),
                name=get_spread_label(pair),
            ),
            row=3,
            col=i + 1,
        )

    title_text = f"{title_text}: {df.index.min().strftime(date_fmt)} - {df.index.max().strftime(date_fmt)}"
    fig.update_layout(
        template="none", **fig_size, title_text=title_text, showlegend=False
    )
    fig.update_yaxes(tickprefix="$")

    # This makes it so the y ranges of the underlying securities in a pair are the same
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


def make_tail_charts(
    pairs: tuple,
    df: pd.DataFrame,
    title_text: str,
    price_col: str = "adj_future",
    date_fmt: str = "%Y-%m-%d",
    fig_size: dict = dict(width=1000, height=500),
) -> go.Figure:
    """Returns figure for side-by-side scatter plots of daily returns overlayed on
    the normal distribution with kurtosis statistics.

    Args:
        pairs: Tuple of two tuples with two str elements, one for each security
            in the pair of the form `{exchange code}_{futures code}_{options_code}`.
        df: Pandas dataframe with the price data to be plotted. Assumes series
            are accessible with a tuple of `{(price_col, security)}`.
        title_text: Text of overall figure.
        price_col: Label of column in df by which the prices of the underlying
            securities are accessible.

    Returns:
        A plotly Figure ready for plotting

    """

    subplot_titles = tuple(get_spread_label(p) for p in pairs)

    fig = make_subplots(rows=1, cols=2, subplot_titles=subplot_titles)

    for i, pair in enumerate(pairs):

        spread = get_spread(pair, df=df, price_col=price_col)
        returns = pd.cut(spread.pct_change(), 100).value_counts().sort_index()
        midpoints = returns.index.map(lambda interval: interval.right).to_numpy()
        norm_dist = norm.pdf(
            midpoints, loc=spread.pct_change().mean(), scale=spread.pct_change().std()
        )

        fig.append_trace(
            go.Scatter(
                x=[interval.mid for interval in returns.index],
                y=returns,
                name=spread.name,
            ),
            row=1,
            col=i + 1,
        )

        fig.append_trace(
            go.Scatter(
                x=[interval.mid for interval in returns.index],
                y=norm_dist,
                name="Normal Distribution",
            ),
            row=1,
            col=i + 1,
        )

    title_text = f"{title_text}: {df.index.min().strftime(date_fmt)} - {df.index.max().strftime(date_fmt)}"
    fig.update_layout(
        template="none", **fig_size, title_text=title_text, showlegend=False
    )
    fig.update_yaxes(tickprefix="")

    for d in fig["data"]:
        xaxis = d["xaxis"]
        xaxis = f"xaxis{xaxis.replace('x', '')}"
        tickvals = d["x"][::3]
        fig.update_layout(
            {xaxis: {"tickmode": "array", "tickvals": tickvals, "tickformat": ".4f"}}
        )

    return fig