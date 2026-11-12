import numpy as np
import pytest

from src.geometry.ransac import ransac, ransac_num_iters


@pytest.mark.parametrize("s,e,expected", [
    # Values from the table in EP3 slide 109 (p = 0.99).
    (2, 0.05, 2), (2, 0.5, 17), (4, 0.3, 17), (4, 0.5, 72),
    (6, 0.4, 97), (8, 0.3, 78), (8, 0.5, 1177),
])
def test_num_iters_matches_course_table(s, e, expected):
    assert ransac_num_iters(0.99, e, s) == expected


def test_ransac_line_fit_with_outliers():
    rng = np.random.default_rng(0)
    n = 200
    xs = rng.uniform(-10, 10, n)
    pts = np.stack([xs, 2.0 * xs + 1.0], axis=1) + rng.normal(scale=0.1, size=(n, 2))
    out = rng.choice(n, size=80, replace=False)          # 40 % outliers
    pts[out] = rng.uniform(-25, 25, size=(80, 2))
    truth = np.ones(n, dtype=bool)
    truth[out] = False

    def fit(idx):
        P = np.c_[pts[idx], np.ones(len(idx))]
        line = np.linalg.svd(P)[2][-1]                   # total least squares
        return line / np.linalg.norm(line[:2])

    def residuals(line):
        return np.abs(np.c_[pts, np.ones(n)] @ line)

    line, mask, _ = ransac(n, fit, residuals, s=2, threshold=0.5, rng=rng)
    slope = -line[0] / line[1]
    intercept = -line[2] / line[1]
    assert abs(slope - 2.0) < 0.02 and abs(intercept - 1.0) < 0.1
    assert (mask & truth).sum() / truth.sum() > 0.95
    assert (mask & ~truth).sum() / max(mask.sum(), 1) < 0.05
