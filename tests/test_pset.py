#!/usr/bin/env python
# -*- coding: utf-8 -*-


from unittest import TestCase

import pandas as pd
from spread_trading.utils import Position, PositionType, Strategy

# This is really just a start and only include a few items used while writing.


class PositionTests(TestCase):
    def test_starting_position(self):
        """Ensure starting position returns correct values"""

        position = Position(
            position_type=PositionType.LONG,
            open_date="2020-10-10",
            security="FCOM",
            shares=100,
            open_price=25.00,
        )

        self.assertEqual(position.open_value, 2500)


class StrategyTests(TestCase):
    def setUp(self):

        df_ticks = pd.DataFrame(
            [
                {"date": "2020-10-10", "adj_close": 25.00, "position_size": 100},
                {"date": "2020-10-11", "adj_close": 25.00, "position_size": 100},
            ],
        )
        df_ticks.date = pd.to_datetime(df_ticks.date)
        self.df_ticks = df_ticks.set_index("date")

        self.strategy = Strategy(
            pair=("SIVR", "SLV"),
            close_threshold=0.0004,
            open_threshold=0.001,
            window=15,
            run=False,
            closed_positions=[],
            df_ticks=self.df_ticks,
        )

        self.strategy.current_date = "2020-10-10"

    def test_open_long_position(self):
        self.strategy.open_long_position("2020-10-10", "SIVR", 100, 25.00)
        self.assertIsInstance(self.strategy.long_position, Position)
        self.assertIsNone(self.strategy.short_position)
        self.assertEqual(self.strategy.start_date, "2020-10-10")

    def test_close_long_position(self):
        self.strategy.open_long_position("2020-10-10", "SIVR", 100, 25.00)
        self.strategy.close_long_position("2020-10-11", 30.00)

        self.assertIsNone(self.strategy.long_position)
        self.assertEqual(self.strategy.closed_positions[0].gross_profit, 500)
        self.assertEqual(self.strategy.closed_positions[0].close_date, "2020-10-11")

    def test_open_short_position(self):
        self.strategy.open_short_position("2020-10-10", "SIVR", 100, 25.00)
        self.assertEqual(len(self.strategy.closed_positions), 0)

        self.assertIsInstance(self.strategy.short_position, Position)
        self.assertEqual(self.strategy.short_position.open_value, 2500)
        self.assertIsNone(self.strategy.long_position)

    def test_close_short_position(self):
        print(len(self.strategy.closed_positions))
        self.strategy.open_short_position("2020-10-10", "SIVR", 100, 25.00)
        self.assertEqual(len(self.strategy.closed_positions), 0)
        self.strategy.close_short_position("2020-10-11", 30.00)

        self.assertEqual(self.strategy.gross_profit, -500)
        self.assertEqual(len(self.strategy.closed_positions), 1)
        self.assertEqual(self.strategy.closed_positions[-1].gross_profit, -500)
        self.assertEqual(self.strategy.closed_positions[-1].close_date, "2020-10-11")