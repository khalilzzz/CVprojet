import numpy as np

from src.geometry.homogeneous import from_h, intersect, line_through, normalize_lines, skew, to_h


def test_to_from_h_roundtrip():
    pts = np.array([[1.0, 2.0], [3.5, -4.0]])
    assert np.allclose(from_h(to_h(pts) * 7.0), pts)


def test_skew_is_cross_product():
    rng = np.random.default_rng(0)
    t, v = rng.normal(size=3), rng.normal(size=3)
    assert np.allclose(skew(t) @ v, np.cross(t, v))
    assert np.allclose(skew(t), -skew(t).T)


def test_line_through_and_intersection():
    l1 = line_through([0.0, 0.0], [1.0, 1.0])      # y = x
    l2 = line_through([0.0, 2.0], [2.0, 0.0])      # x + y = 2
    p = intersect(l1, l2)
    assert np.allclose(p[:2] / p[2], [1.0, 1.0])


def test_parallel_lines_meet_at_infinity():
    l1 = line_through([0.0, 0.0], [1.0, 0.0])
    l2 = line_through([0.0, 1.0], [1.0, 1.0])
    p = intersect(l1, l2)
    assert abs(p[2]) < 1e-12 and abs(p[1]) < 1e-12


def test_normalized_line_gives_point_distance():
    l = normalize_lines(line_through([0.0, 0.0], [10.0, 0.0]))
    assert np.isclose(abs(l @ np.array([3.0, 5.0, 1.0])), 5.0)
