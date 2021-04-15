#!/usr/bin/env python
# -*- coding: utf-8 -*-

import datetime
from unittest import TestCase

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
    def test_open_long_position(self):

        strategy = Strategy(
            open_threshold=0.001,
            close_threshold=0.0004,
            window=15,
            pair=("SIVR", "SLV"),
            cash=10000,
        )

        self.assertIsNone(strategy.start_date)
        strategy.open_long_position("2020-10-10", "SIVR", 100, 25.00)
        self.assertIsInstance(strategy.long_position, Position)
        self.assertIsNone(strategy.short_position)
        self.assertEqual(strategy.start_date, "2020-10-10")
        self.assertEqual(strategy.cash, 7500)

        del strategy

    def test_close_long_position(self):

        strategy = Strategy(
            open_threshold=0.001,
            close_threshold=0.0004,
            window=15,
            pair=("SIVR", "SLV"),
            cash=10000,
        )

        strategy.open_long_position("2020-10-10", "SIVR", 100, 25.00)
        strategy.close_long_position("2020-10-11", 30.00)

        self.assertIsNone(strategy.long_position)
        self.assertEqual(strategy.cash, 10000 - 2500 + 3000)
        self.assertEqual(strategy.closed_positions[0].gross_profit, 500)
        self.assertEqual(strategy.closed_positions[0].close_date, "2020-10-11")

        del strategy

    def test_open_short_position(self):

        strategy = Strategy(
            open_threshold=0.001,
            close_threshold=0.0004,
            window=15,
            pair=("SIVR", "SLV"),
            cash=10000,
        )

        strategy.open_short_position("2020-10-10", "SIVR", 100, 25.00)
        self.assertEqual(len(strategy.closed_positions), 0)

        self.assertIsInstance(strategy.short_position, Position)
        self.assertEqual(strategy.short_position.open_value, 2500)
        self.assertIsNone(strategy.long_position)
        self.assertEqual(strategy.cash, 10000)

        del strategy

    def test_close_short_position(self):

        strategy = Strategy(
            pair=("SIVR", "SLV"),
            open_threshold=0.001,
            close_threshold=0.0004,
            window=15,
            cash=10000,
        )

        print(len(strategy.closed_positions))
        strategy.open_short_position("2020-10-10", "SIVR", 100, 25.00)
        self.assertEqual(len(strategy.closed_positions), 0)
        strategy.close_short_position("2020-10-11", 30.00)

        self.assertEqual(strategy.cash, 9500)
        self.assertEqual(strategy.gross_profit, -500)
        self.assertEqual(len(strategy.closed_positions), 1)
        self.assertEqual(strategy.closed_positions[-1].gross_profit, -500)
        self.assertEqual(strategy.closed_positions[-1].close_date, "2020-10-11")