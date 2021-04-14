#!/usr/bin/env python
# -*- coding: utf-8 -*-

from unittest import TestCase

from spread_trading import utils

# This is really just a start and only include a few items used while writing.


class FetchDataTests(TestCase):
    def test_security_code(self):
        """Ensure code for security is properly formatted."""

        sec_code = utils.get_security_code("CBT_FV_FV", "OWF", "H2019", "IVM", [1])
        test_sec_code = ("OWF/CBT_FV_FV_H2019_IVM", {"column_index": [1]})

        self.assertEqual(sec_code, test_sec_code)


class PrepDataTests(TestCase):
    def test_parse_column_label(self):
        """Ensure column label is parsed correctly."""

        parsed_col = utils.parse_column_label("OWF/CBT_FV_FV_H2019_IVM - Futures")
        test_parsed_col = ("OWF", "CBT_FV_FV", "H2019", "IVM", "Futures")

        self.assertEqual(parsed_col, test_parsed_col)
