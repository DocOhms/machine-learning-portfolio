# PUBLIC REVIEW: computational expressions retained; provenance comments added.
# Formula explanations and package boundaries: docs/CALCULATIONS.md.
# Full function/source index: docs/FUNCTION_INDEX.md.
"""Regression checks for display dates; forecast keys/features are unchanged."""
from pathlib import Path
import sys
import unittest
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from common_io import epi_week_start


class MMWRCalendarTests(unittest.TestCase):
    # Calculation provenance:
    #   epi_week_start -> common_io.epi_week_start (src/common_io.py)
    #   pd.Series -> pandas.Series (third-party package)
    #   pd.testing.assert_series_equal -> pandas.testing.assert_series_equal (third-party package)
    #   pd.to_datetime -> pandas.to_datetime (third-party package)
    def test_week_53_and_year_boundary(self):
        years = pd.Series([2025, 2025, 2026, 2026, 2019, 2020])
        weeks = pd.Series([1, 53, 1, 53, 40, 39])
        dates = epi_week_start(years, weeks)
        expected = ['2024-12-29', '2025-12-28', '2026-01-04', None,
                    '2019-09-29', '2020-09-20']
        pd.testing.assert_series_equal(dates, pd.to_datetime(pd.Series(expected)))

    # Calculation provenance:
    #   epi_week_start -> common_io.epi_week_start (src/common_io.py)
    #   pd.Series -> pandas.Series (third-party package)
    #   pd.Timestamp -> pandas.Timestamp (third-party package)
    #   pd.isna -> pandas.isna (third-party package)
    def test_index_and_invalid_values(self):
        dates = epi_week_start(pd.Series([2025, None], index=['a', 'b']),
                               pd.Series([53, 1], index=['a', 'b']))
        self.assertEqual(dates.loc['a'], pd.Timestamp('2025-12-28'))
        self.assertTrue(pd.isna(dates.loc['b']))


if __name__ == '__main__':
    unittest.main()
