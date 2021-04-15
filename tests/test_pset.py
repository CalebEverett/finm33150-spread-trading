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
            securities=("SIVR", "SLV"),
            cash=10000,
        )

        self.assertIsNone(strategy.start_date)
        strategy.open_long_position("2020-10-10", "SIVR", 100, 25.00)
        self.assertEqual(strategy.start_date, "2020-10-10")
        self.assertEqual(strategy.cash, 7500)

    def test_close_long_position(self):

        strategy = Strategy(
            open_threshold=0.001,
            close_threshold=0.0004,
            window=15,
            securities=("SIVR", "SLV"),
            cash=10000,
        )

        strategy.open_long_position("2020-10-10", "SIVR", 100, 25.00)
        strategy.close_long_position("2020-10-10", 30.00)

        self.assertIsNone(strategy.long_position)
        self.assertEqual(strategy.cash, 10000 - 2500 + 3000)
        self.assertEqual(strategy.closed_positions[0].profit, 500)


# class TradeTests(TestCase):
#     def test_trade_signs(self):
#         """Shares are positive and value is negative for long trade."""

#         trade = Trade(
#             date_str="2020-10-10",
#             trade_type=TradeType.OPEN_LONG,
#             security="FCOM",
#             shares=100,
#             adj_close=25.00,
#         )

#         self.assertTrue(trade.trade_shares > 0)
#         self.assertTrue(trade.trade_value < 0)

#         trade.trade_type = TradeType.OPEN_SHORT
#         self.assertTrue(trade.trade_shares < 0)
#         self.assertTrue(trade.trade_value > 0)

#         trade.trade_type = TradeType.CLOSE_LONG
#         self.assertTrue(trade.trade_shares < 0)
#         self.assertTrue(trade.trade_value > 0)

#         trade.trade_type = TradeType.CLOSE_SHORT
#         self.assertTrue(trade.trade_shares > 0)
#         self.assertTrue(trade.trade_value < 0)


# class OldPositionTests(TestCase):
#     def test_starting_position(self):
#         """Ensure starting position returns correct values"""

#         position = Position(date_str="2020-10-10", security="FCOM", cash=100)

#         self.assertEqual(position.adj_close, 0)
#         self.assertEqual(position.date_str, "2020-10-10")

#     def test_new_long_open(self):
#         """Ensure position values are updated correctly for a new open long trade."""

#         position = Position(date_str="2020-10-10", security="FCOM", cash=10000)

#         trade = Trade(
#             date_str="2020-10-11",
#             trade_type=TradeType.OPEN_LONG,
#             security="FCOM",
#             shares=100,
#             adj_close=25.00,
#         )

#         new_position = position(trade)

#         self.assertEqual(new_position.date_str, "2020-10-11")
#         self.assertEqual(new_position.security, "FCOM")
#         self.assertEqual(new_position.adj_close, 25.00)
#         self.assertEqual(new_position.shares_long, 100)
#         self.assertEqual(new_position.shares_short, 0)
#         self.assertEqual(new_position.basis_long, -trade.trade_value)
#         self.assertEqual(new_position.basis_short, 0)
#         self.assertEqual(new_position.cash, position.cash + trade.trade_value)
#         self.assertEqual(
#             new_position.market_value_long,
#             new_position.adj_close * new_position.shares_long,
#         )

#     def test_new_open_short(self):
#         """Ensure position values are updated correctly for a new open short trade."""

#         position = Position(date_str="2020-10-10", security="FCOM", cash=10000)

#         trade = Trade(
#             date_str="2020-10-11",
#             trade_type=TradeType.OPEN_SHORT,
#             security="FCOM",
#             shares=100,
#             adj_close=25.00,
#         )

#         new_position = position(trade)

#         self.assertEqual(new_position.shares_long, 0)
#         self.assertEqual(new_position.shares_short, 100)
#         self.assertEqual(new_position.basis_short, -trade.trade_value)
#         self.assertEqual(new_position.basis_long, 0)
#         self.assertEqual(new_position.cash, position.cash + trade.trade_value)
#         self.assertEqual(
#             new_position.market_value_short,
#             new_position.adj_close * new_position.shares_short * -1,
#         )

#     def test_new_close_short(self):
#         """Ensure position values are updated correctly for a new open short trade."""

#         position = Position(
#             date_str="2020-10-10",
#             security="FCOM",
#             shares_short=100,
#             cash=12500,
#             basis_short=-2500,
#         )

#         trade = Trade(
#             date_str="2020-10-11",
#             trade_type=TradeType.CLOSE_SHORT,
#             security="FCOM",
#             shares=100,
#             adj_close=25.00,
#         )

#         new_position = position(trade)

#         self.assertEqual(new_position.shares_long, 0)
#         self.assertEqual(new_position.shares_short, 0)
#         self.assertEqual(new_position.basis_short, 0)
#         self.assertEqual(new_position.basis_long, 0)
#         self.assertEqual(new_position.cash, position.cash + trade.trade_value)
#         self.assertEqual(
#             new_position.market_value_short,
#             new_position.adj_close * new_position.shares_short * -1,
#         )
