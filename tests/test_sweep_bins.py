"""Bin construction has to cover the requested range without silently dropping
the hardest slice -- a truncated trailing bin is still a valid measurement."""

from __future__ import annotations

import pytest

from scripts.sweep_snr_wer import build_bins


def test_exact_division() -> None:
    assert build_bins(-8.0, 6.0, 2.0) == [
        (-8.0, -6.0), (-6.0, -4.0), (-4.0, -2.0), (-2.0, 0.0),
        (0.0, 2.0), (2.0, 4.0), (4.0, 6.0),
    ]


def test_trailing_partial_bin_is_kept_and_clipped() -> None:
    bins = build_bins(-8.0, 5.0, 2.0)
    assert bins[-1] == (4.0, 5.0)
    assert bins[0][0] == -8.0


def test_bins_are_contiguous_and_cover_the_range() -> None:
    bins = build_bins(-8.0, 6.0, 1.5)
    assert bins[0][0] == -8.0
    assert bins[-1][1] == 6.0
    for (_, upper), (lower, _) in zip(bins, bins[1:]):
        assert upper == lower


@pytest.mark.parametrize(("lo", "hi", "width"), [(0.0, 6.0, 0.0), (0.0, 6.0, -1.0)])
def test_non_positive_width_rejected(lo: float, hi: float, width: float) -> None:
    with pytest.raises(ValueError, match="bin-width"):
        build_bins(lo, hi, width)


@pytest.mark.parametrize(("lo", "hi"), [(6.0, 6.0), (6.0, 0.0)])
def test_empty_or_inverted_range_rejected(lo: float, hi: float) -> None:
    with pytest.raises(ValueError, match="max-snr"):
        build_bins(lo, hi, 2.0)
