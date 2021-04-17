from dataclasses import dataclass
import datetime
import hashlib
import os
from enum import Enum
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import quandl
from plotly import colors
from plotly.subplots import make_subplots
from scipy import stats
from tabulate import tabulate

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


def parse_column_label_eod(c: str) -> tuple:
    """Parses a column label as returned by Quandl into individual data elements
    to facilitate the flattening of the data.

    Args:
        c: Column label as returned by Quandl.

    Returns:
        Tuple with the following elements:
            feed: Data feed code, i.e., 'EOD'
            sec: Security of the form `{exchange_code}_{options_code}_{futures_code}`
            series: Label of the data series, i.e., 'Futures`

    """

    elems = c.split("/")
    feed = elems[0]
    sec, series = elems[1].split(" - ")

    return tuple([feed, sec, series])


def parse_column_label_owf(c: str) -> tuple:
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


def get_spread_label_owf(pair: tuple) -> str:
    """Return labels for spread between two securities in the form
    `{exchange code}:{security_2 futures code}-{security_1 futures code}`

    Args:
        pairs: Tuples with two str elements, one for each security
            in the form `{exchange code}_{futures code}_{options_code}`.

    Returns:
        String with label for spread between two securities.
    """

    return f"{pair[0].split('_')[0]}:{pair[1].split('_')[-1]}-{pair[0].split('_')[-1]}"


def get_spread_label_eod(pair: tuple) -> str:
    """Return label for a pair for EOD securities

    Args:
        pairs: Tuples of two security strings.

    Returns:
        String with label for spread between two securities.
    """

    return f"{pair[1]}-{pair[0]}"


FEED_PARAMS = {
    "OWF": {
        "labels": ["data_feed", "security", "expiration", "model", "series"],
        "parse_fn": parse_column_label_owf,
        "label_fn": get_spread_label_owf,
    },
    "EOD": {
        "labels": ["data_feed", "security", "series"],
        "parse_fn": parse_column_label_eod,
        "label_fn": get_spread_label_eod,
    },
}


def expand_series(
    s: pd.Series, data_feed: str, date_fmt: str = "%Y-%m-%d"
) -> pd.DataFrame:
    """Expand original series returned from Quandl to include seprate
    series for each data element as returned by `parse_column_label.

    Args:
        s: Original series returned from Quandl.
        data_feed: Quandle data feed code.
        date_fmt: String formatting for the date index.

    Returns
        Dataframe with one column for each data element, including
        the original series.
    """

    labels = FEED_PARAMS[data_feed]["labels"]
    parse_fn = FEED_PARAMS[data_feed]["parse_fn"]

    values = parse_fn(s.name)

    df_exp = pd.DataFrame({"date": s.index.values})
    for i, label in enumerate(labels):
        df_exp[label] = values[i]

    df_exp["value"] = s.values
    df_exp["series"] = df_exp["series"].str.lower()

    return df_exp


class ReturnType(Enum):
    LOG: str = "log"
    SIMPLE: str = "simple"
    DIFF: str = "diff"


def get_returns(
    df: pd.DataFrame,
    return_type: ReturnType = "log",
) -> pd.Series:
    """Calculates return on a security.

    Args:
        df: Pandas dataframe with the prices used to calculate returns.
        return_type: Either `log`, `simple` or `diff` to specify how returns
            are calculated.

    Returns:
        Pandas series of return.
    """

    if return_type is not None:
        return_type = ReturnType(return_type)

    if return_type == ReturnType.LOG:
        returns = (df / df.shift(1)).apply(np.log).dropna()
    elif return_type == ReturnType.SIMPLE:
        returns = df.pct_change().dropna()
    elif return_type == ReturnType.DIFF:
        returns = df.diff().dropna()

    returns.columns = pd.MultiIndex.from_tuples(
        tuples=[(f"adj_return", security) for security in returns.columns],
        names=["series", "security"],
    )

    return returns


def get_spread(
    pair: tuple,
    df: pd.DataFrame,
    data_feed: str = "EOD",
    price_col: str = "adj_close",
    return_type: ReturnType = "log",
) -> pd.Series:
    """Calculates spread between two series. The spread will be expressed
    as the second security of the pair less the first one.

    Args:
        pair: Tuple of two str elements, one for each security
        df: Pandas dataframe with the price data to be plotted. Assumes series
            are accessible with a tuple of `{(price_col, security)}`.
        price_col: Label of column in df by which the prices of the underlying
            securities are accessible.
        data_feed: Quandl data feed code.
        return_type: Either `log`, `simple` or `diff` to specify how returns
            are calculated.

    Returns:
        Pandas series of spread.
    """

    label = FEED_PARAMS[data_feed]["label_fn"](pair)
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


# =============================================================================
# Strategy
# =============================================================================


class PositionType(Enum):
    LONG: str = "long"
    SHORT: str = "short"


@dataclass
class Position:
    position_type: PositionType
    open_date: str
    security: str
    shares: int
    open_price: float
    open_transact_cost: float = 0
    close_price: float = None
    close_transact_cost: float = 0
    closed: bool = False
    close_date: str = None
    transact_cost_per_share: float = 0.01

    def __post_init__(self):
        self.open_transact_cost = self.shares * self.transact_cost_per_share

    @property
    def open_value(self):
        return self.shares * self.open_price

    @property
    def close_value(self):
        return self.shares * self.close_price

    @property
    def transact_cost(self):
        return self.open_transact_cost + self.close_transact_cost

    @property
    def gross_profit(self):
        if self.closed:
            if self.position_type == PositionType.LONG:
                profit = self.close_value - self.open_value
            else:
                profit = self.open_value - self.close_value
            return profit
        else:
            raise Exception("Position is still open.")

    @property
    def net_profit(self):
        if self.closed:
            return self.gross_profit - self.transact_cost
        else:
            raise Exception("Position is still open.")

    def unrealized_profit(self, current_price: float):
        if self.closed:
            raise Exception("Position is closed.")
        else:
            if self.position_type == PositionType.LONG:
                mv = self.shares * (current_price - self.open_price)
            else:
                mv = self.shares * (self.open_price - current_price)
            return mv - self.shares * self.transact_cost_per_share

    def close(self, close_date: str, close_price: float):
        self.close_date = close_date
        self.close_price = close_price
        self.close_transact_cost = self.shares * self.transact_cost_per_share
        self.closed = True


class Strategy:
    """Class for conducting a backtest of a spread trading strategy.

    Properties:
        pair: Tuple of securities
        ticks: Pandas DataFrame with `adj_close`, `position_size`, and `adj_returns`
            columns.
        open_threshold: Absolute value of difference in returns above which
            a position will be opened if one is not already open.
        close_threshold: Absolute value of difference in returns above which
            a position will be opened if one is not already open.
        window: Trailing number of days over which the returns for
            purposes of determining the spread between the two securities were
            calculated.
        cash: Balance of cash on hand
        gross_profit: Realized gross profit as of `current_date`
        transact_cost: Cash transaction costs incurred as of `current_date`
        start_date: Starting date of the strategy
        current_date: Current date of strategy
        end_date: str = Ending date of strategy

        long_position = Open long position as of `current_date` if any
        short_position = Open short position as of `current_date` if any
        closed_positions = Closed positions as of `current date`
    """

    def __init__(
        self,
        pair: Tuple,
        df_ticks: pd.DataFrame,
        open_threshold: float,
        close_threshold: float,
        window: int,
        closed_positions: list,
        run: bool = True,
        transact_cost_per_share: float = 0.01,
    ):

        self.pair = pair
        self.df_ticks = df_ticks
        self.open_threshold = open_threshold
        self.close_threshold = close_threshold
        self.window = window
        self.gross_profit: float = 0
        self.transact_cost: float = 0
        self.current_date: str = None
        self.long_position: Position = None
        self.short_position: Position = None
        self.closed_positions = closed_positions
        self.transact_cost_per_share = transact_cost_per_share

        self.start_date = self.df_ticks.index.min().strftime("%Y-%m-%d")
        self.end_date = self.df_ticks.index.max().strftime("%Y-%m-%d")

        self.capital = (
            self.df_ticks["position_size"] * self.df_ticks["adj_close"]
        ).max().max() * 2

        # Frequency of `BM` is last business day of month to make sure
        # positions are closed if the calendar last day of the month
        # is not a trading day.
        self.month_ends = (
            pd.date_range(self.start_date, self.end_date, freq="BM")
            .strftime("%Y-%m-%d")
            .to_list()
        )

        if run:
            self.run()

    @property
    def net_profit(self):
        return self.gross_profit - self.transact_cost

    @property
    def unrealized_profit(self):
        if self.long_position:
            return self.long_position.unrealized_profit(
                self.current_prices[self.long_position.security]
            ) + self.short_position.unrealized_profit(
                self.current_prices[self.short_position.security]
            )
        else:
            return 0

    def open_long_position(
        self, open_date: str, security: str, shares: int, open_price
    ):
        if self.long_position is not None:
            raise Exception("An open long position already exists.")

        if security not in self.pair:
            raise Exception(
                f"{security} is not included in strategy securities:"
                f" {str(self.pair)}"
            )

        if open_date < self.current_date:
            raise Exception(
                f"Position open date of {open_date} is before strategy current"
                f"date of {self.current_date}"
            )

        self.long_position = Position(
            position_type=PositionType.LONG,
            open_date=open_date,
            security=security,
            shares=shares,
            open_price=open_price,
            transact_cost_per_share=self.transact_cost_per_share,
        )

        self.transact_cost += self.long_position.open_transact_cost

    def open_short_position(
        self, open_date: str, security: str, shares: int, open_price
    ):
        if self.short_position is not None:
            raise Exception("An open short position already exists.")

        self.short_position = Position(
            position_type=PositionType.SHORT,
            open_date=open_date,
            security=security,
            shares=shares,
            open_price=open_price,
            transact_cost_per_share=self.transact_cost_per_share,
        )

        self.transact_cost += self.short_position.open_transact_cost

    def close_long_position(self, close_date: str, close_price: float):
        if self.long_position is None:
            raise Exception("There is no open long position.")

        self.long_position.close(close_date, close_price)

        self.gross_profit += self.long_position.gross_profit
        self.transact_cost += self.long_position.close_transact_cost

        self.closed_positions.append(self.long_position)
        self.long_position = None

    def close_short_position(self, close_date: str, close_price: float):
        if self.short_position is None:
            raise Exception("There is no open long short.")

        self.short_position.close(close_date, close_price)

        self.gross_profit += self.short_position.gross_profit
        self.transact_cost += self.short_position.close_transact_cost

        self.closed_positions.append(self.short_position)
        self.short_position = None

    def record_trade(
        self,
        open_position: bool,
        date: str,
        spread: int,
        long_security: str,
        short_security: str,
    ):
        if open_position:
            if short_security == self.pair[1]:
                trade_type = "short"
            else:
                trade_type = "buy"
        else:
            if short_security == self.pair[1]:
                trade_type = "buy"
            else:
                trade_type = "short"

        self.trades.append(
            {
                "date": date,
                "trade_type": trade_type,
                "spread": spread,
                "long_security": long_security,
                "short_security": short_security,
                "open_position": open_position,
            }
        )

    def run(self):
        self.stats = []
        for tick in self.df_ticks.iterrows():
            date, tick = tick
            spread = tick.spread[0]
            self.current_date = date.strftime("%Y-%m-%d")
            self.current_prices = tick.adj_close
            returns = tick.rolling_adj_return.sort_values()
            short_security = returns.index[-1]
            long_security = returns.index[0]

            # Just testing long position since both long and short positions
            # are always open or None. Don't open a position on the last day
            # of the strategy.

            # Closing positions first so that opening logic works whether opening
            # from not having an open position or from after having sold because the
            # spread reversed.

            if self.long_position is not None and (
                (
                    abs(spread) < self.close_threshold
                    or self.current_date in self.month_ends
                    or self.current_date == self.end_date
                )
                or self.long_position.security != long_security
            ):

                self.close_long_position(
                    close_date=self.current_date,
                    close_price=tick.adj_close[self.long_position.security],
                )

                self.close_short_position(
                    close_date=self.current_date,
                    close_price=tick.adj_close[self.short_position.security],
                )

            if (
                abs(spread) > self.open_threshold
                and self.long_position is None
                and self.current_date != self.end_date
                and self.current_date not in self.month_ends
            ):

                self.open_long_position(
                    open_date=self.current_date,
                    security=long_security,
                    shares=tick.position_size[long_security],
                    open_price=tick.adj_close[long_security],
                )

                self.open_short_position(
                    open_date=self.current_date,
                    security=short_security,
                    shares=tick.position_size[short_security],
                    open_price=tick.adj_close[short_security],
                )

            if self.current_date == self.start_date:
                prior_total_profit = 0
            else:
                prior_total_profit = self.stats[-1]["total_profit"]

            total_profit = self.net_profit + self.unrealized_profit
            tick_profit = total_profit - prior_total_profit
            total_return = np.log(total_profit + self.capital) - np.log(self.capital)
            tick_return = np.log(tick_profit + self.capital) - np.log(self.capital)
            self.stats.append(
                {
                    "date": date,
                    "realized_profit": self.net_profit,
                    "unrealized_profit": self.unrealized_profit,
                    "total_profit": total_profit,
                    "tick_profit": tick_profit,
                    "total_return": total_return,
                    "tick_return": tick_return,
                }
            )

    def get_day_trades(self, date: str):
        trades = []
        headers = ["trans", "sec", "shrs", "price", "profit"]
        opened_positions = [p for p in self.closed_positions if p.open_date == date]
        closed_positions = [p for p in self.closed_positions if p.close_date == date]

        for p in opened_positions:
            trans = "buy" if p.position_type.value == "long" else "short"
            trades.append([trans, p.security, p.shares, p.open_price, -p.transact_cost])

        for p in closed_positions:
            trans = "sell" if p.position_type.value == "long" else "cover"
            trades.append([trans, p.security, p.shares, p.close_price, p.net_profit])

        hover_text = tabulate(
            trades,
            tablefmt="plain",
            headers=headers,
            floatfmt=("", "", ",.0f", ",.2f", ",.0f"),
        ).replace("\n", "<br>")

        if opened_positions:
            long_position = [
                p for p in opened_positions if p.position_type.value == "long"
            ][0]
            if long_position.security == self.pair[0]:
                trade_type = "short"
            else:
                trade_type = "buy"
        elif closed_positions:
            long_position = [
                p for p in closed_positions if p.position_type.value == "long"
            ][0]
            if long_position.security == self.pair[0]:
                trade_type = "buy"
            else:
                trade_type = "short"

        size = 6.5 if opened_positions and closed_positions else 4.5

        return (
            date,
            self.df_ticks.loc[date].spread.values[0],
            trade_type,
            size,
            hover_text,
        )

    def plot(
        self,
        title_text: str = "Spread Trading Chart",
        data_feed: str = "EOD",
        date_fmt: str = "%Y-%m-%d",
    ) -> go.Figure:
        """Returns figure for side-by-side q-q plots of daily returns.

        Args:
            title: Figure title.
            data_feed: Quandl data feed code.
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

        dates = self.df_ticks.index.get_level_values("date")
        start_date = dates.min()
        end_date = dates.max()
        date_range = pd.date_range(start_date, end_date, freq="D")
        range_breaks = date_range[~date_range.isin(dates)]

        label_fn = FEED_PARAMS[data_feed]["label_fn"]

        fig = go.Figure()

        fig = make_subplots(
            rows=2,
            cols=1,
            subplot_titles=[
                "Trades",
                f"Returns: Total = {self.stats[-1]['total_return']:0.4f}, ${self.stats[-1]['total_profit']:0.0f}",
            ],
            shared_xaxes=True,
            vertical_spacing=0.10,
            specs=[[dict(secondary_y=True)], [dict(secondary_y=True)]],
        )

        # =======================
        # Spread
        # =======================

        fig.append_trace(
            go.Scatter(
                y=self.df_ticks["spread"],
                x=dates,
                name="spread",
                line=dict(width=1),
            ),
            row=1,
            col=1,
        )

        def add_band(positive: int = 1):
            fig.append_trace(
                go.Scatter(
                    y=[self.close_threshold * positive] * len(dates),
                    x=dates,
                    name="close_threshold",
                    line=dict(width=0),
                    hoverinfo="skip",
                    showlegend=False,
                ),
                row=1,
                col=1,
            )

            fig.append_trace(
                go.Scatter(
                    y=[self.open_threshold * positive] * len(dates),
                    x=dates,
                    name="open_threshold",
                    fill="tonexty",
                    line=dict(width=0),
                    line_color=COLORS[4],
                    hoverinfo="skip",
                    showlegend=False,
                ),
                row=1,
                col=1,
            )

        add_band()
        add_band(-1)

        # =======================
        # Trades
        # =======================

        trade_dates = sorted(
            sum([[p.open_date, p.close_date] for p in self.closed_positions], [])
        )

        df_trades = pd.DataFrame(
            map(self.get_day_trades, trade_dates),
            columns=["date", "spread", "trans", "marker_size", "text"],
        )

        fig.append_trace(
            go.Scatter(
                y=df_trades.spread,
                x=df_trades.date,
                name="trades",
                mode="markers",
                marker=dict(
                    color=df_trades.trans.map({"buy": "green", "short": "red"}),
                    size=df_trades.marker_size,
                    line=dict(width=0),
                ),
                text=df_trades.text,
                hovertemplate="%{text}",
            ),
            row=1,
            col=1,
        )

        # =======================
        # Returns
        # =======================

        df_stats = pd.DataFrame(self.stats)
        fig.add_trace(
            go.Scatter(
                y=df_stats["total_return"],
                x=df_stats["date"],
                name="total_return",
                line=dict(width=1),
                line_color=COLORS[2],
            ),
            secondary_y=False,
            row=2,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                y=df_stats["tick_return"],
                x=df_stats["date"],
                name="tick_return",
                line=dict(width=1),
                line_color=COLORS[1],
            ),
            row=2,
            col=1,
            secondary_y=True,
        )

        # =======================
        # Figure
        # =======================

        title_text = (
            f"{title_text}: {label_fn(self.pair)}: {start_date.strftime(date_fmt)}"
            f" - {end_date.strftime(date_fmt)}"
        )

        fig.update_layout(
            template="none",
            autosize=True,
            height=800,
            title_text=title_text,
        )

        shapes = []
        for month_end in self.month_ends:
            shapes.append(
                dict(
                    type="line",
                    yref="paper",
                    y0=0.55,
                    y1=0.98,
                    xref="x",
                    line_dash="dot",
                    line_width=1,
                    x0=month_end,
                    x1=month_end,
                )
            )
            shapes.append(
                dict(
                    type="line",
                    yref="paper",
                    y0=0.05,
                    y1=0.42,
                    xref="x",
                    line_dash="dot",
                    line_width=1,
                    x0=month_end,
                    x1=month_end,
                )
            )

        fig.update_layout(
            shapes=shapes,
            hoverlabel=dict(font_family="Courier New, monospace"),
            # hovermode="x unified",
        )
        fig.update_xaxes(rangebreaks=[dict(values=range_breaks)])

        returns_annotation = get_moments_annotation(
            df_stats.tick_return,
            xref="paper",
            yref="paper",
            x=1,
            y=0.4,
            xanchor="left",
        )
        fig.add_annotation(returns_annotation)

        spread_annotation = get_moments_annotation(
            self.df_ticks.spread,
            xref="paper",
            yref="paper",
            x=1,
            y=0.8,
            xanchor="left",
        )
        fig.add_annotation(spread_annotation)

        return fig


def get_ticks(pair: Tuple, df: pd.DataFrame, window: int):
    """Creates table of spread, returns, closing prices and trade amounts to be processed
    iteratively by a Strategy instance.
    """

    columns = ["adj_close", "adj_return", "med_dollar_volume", "volume"]
    df_ticks = df.copy().iloc[
        :,
        (
            df.columns.get_level_values(0).isin(columns)
            & df.columns.get_level_values(1).isin(pair)
        ),
    ]

    less_liquid = df_ticks["med_dollar_volume"].sum(axis=0).sort_values().index[0]
    dollar_position_size = df_ticks[("med_dollar_volume", less_liquid)] / 100

    for S in pair:
        df_ticks[("position_size", S)] = (
            dollar_position_size / df_ticks[("adj_close", S)]
        ).round()

    for S in pair:
        df_ticks[("rolling_adj_return", S)] = (
            df_ticks[("adj_return", S)].rolling(window).sum()
        )

    df_ticks["spread"] = (
        df_ticks[("rolling_adj_return", pair[1])]
        - df_ticks[("rolling_adj_return", pair[0])]
    )

    return df_ticks.dropna()


# =============================================================================
# Charts
# =============================================================================

COLORS = colors.qualitative.D3


def make_spread_charts(
    pairs: tuple,
    df: pd.DataFrame,
    title_text: str,
    data_feed: str = "EOD",
    price_col: str = "adj_future",
    date_fmt: str = "%Y-%m-%d",
    height: int = 600,
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

    label_fn = FEED_PARAMS[data_feed]["label_fn"]

    subplot_titles = (pairs[0][1], pairs[1][1], pairs[0][0], pairs[1][0])
    subplot_titles += tuple(label_fn(p) for p in pairs)

    fig = make_subplots(rows=3, cols=2, subplot_titles=subplot_titles)

    y_ranges = {}

    positions = [
        (pairs[0][1], 1, 1),
        (pairs[0][0], 2, 1),
        (pairs[1][1], 1, 2),
        (pairs[1][0], 2, 2),
    ]

    for security, row, col in positions:
        series = df[(price_col, security)].dropna()
        fig.append_trace(
            go.Scatter(x=series.index, y=series, name=security),
            row=row,
            col=col,
        )

    for i, pair in enumerate(pairs):
        spread = get_spread(pair, df=df, price_col=price_col)
        fig.append_trace(
            go.Scatter(
                x=spread.index,
                y=spread,
                name=label_fn(pair),
            ),
            row=3,
            col=i + 1,
        )

    beg_date = df.index.min().strftime(date_fmt)
    end_date = df.index.max().strftime(date_fmt)
    title_text = f"{title_text}: {beg_date} - {end_date}"

    fig.update_layout(
        template="none",
        autosize=True,
        height=height,
        title_text=title_text,
        showlegend=False,
    )
    fig.update_yaxes(tickprefix="$")

    return fig


def get_moments_annotation(
    s: pd.Series, xref: str, yref: str, x: float, y: float, xanchor: str
) -> go.layout.Annotation:
    """Calculates summary statistics for a series and returns and
    Annotation object.
    """

    labels = [
        ("obs", lambda x: f"{x:>16d}"),
        ("min:max", lambda x: f"{x[0]:>0.3f}:{x[1]:>0.3f}"),
        ("mean", lambda x: f"{x:>13.3e}"),
        ("std", lambda x: f"{x:>15.3e}"),
        ("skewness", lambda x: f"{x:>11.2f}"),
        ("kurtosis", lambda x: f"{x:>13.2f}"),
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
    data_feed: str = "EOD",
    date_slices: Tuple[slice, slice] = None,
    price_col: str = "adj_future",
    date_fmt: str = "%Y-%m-%d",
    height: int = 450,
    tick_skip: int = 4,
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
        data_feed: Quandl data feed code.
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

    label_fn = FEED_PARAMS[data_feed]["label_fn"]

    subplot_titles = tuple(label_fn(p) for p in pairs)

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
        autosize=True,
        height=height,
        title_text=title_text,
        showlegend=False,
    )
    fig.update_yaxes(tickprefix="%")

    for d in fig["data"]:
        xaxis = d["xaxis"]
        xaxis = f"xaxis{xaxis.replace('x', '')}"
        tickvals = d["x"][::tick_skip]
        fig.update_layout(
            {xaxis: {"tickmode": "array", "tickvals": tickvals, "tickformat": ".4f"}}
        )

    return fig
