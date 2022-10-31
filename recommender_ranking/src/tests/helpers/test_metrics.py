import pytest

from src.gid_ml_framework.helpers.metrics import ap_at_k, map_at_k


# https://github.com/benhamner/Metrics/blob/master/Python/ml_metrics/test/test_average_precision.py
# unittest -> pytest
class TestAveragePrecision:
    def test_ap_k(self):
        result_ap_at_k = ap_at_k(range(1, 6), [6, 4, 7, 1, 2], k=2)
        assert pytest.approx(result_ap_at_k) == 0.25

        result_ap_at_k = ap_at_k(range(1, 6), [1, 1, 1, 1, 1], k=5)
        assert pytest.approx(result_ap_at_k) == 0.2

        predicted = list(range(1, 21))
        predicted.extend(range(200, 600))
        result_ap_at_k = ap_at_k(range(1, 100), predicted, k=20)
        assert pytest.approx(result_ap_at_k) == 1.0

    def test_ap_k_order_same_result(self):
        result_ap_at_k = ap_at_k(range(1, 6), [1, 2, 4, 6, 7], k=2)
        result_changed_order = ap_at_k(range(1, 6), [2, 1, 4, 6, 7], k=2)
        assert pytest.approx(result_ap_at_k) == result_changed_order

        result_ap_at_k = ap_at_k(range(1, 6), [1, 2, 4, 6, 7], k=2)
        result_changed_order = ap_at_k(range(1, 6), [1, 2, 6, 7, 4], k=2)
        assert pytest.approx(result_ap_at_k) == result_changed_order

    def test_ap_k_order_different_result(self):
        result_ap_at_k = ap_at_k(range(1, 6), [4, 6, 7, 1, 2], k=5)
        result_changed_order = ap_at_k(range(1, 6), [6, 4, 7, 2, 1], k=5)
        assert pytest.approx(result_ap_at_k) != result_changed_order

        result_ap_at_k = ap_at_k(range(1, 6), [1, 2, 4, 6, 7], k=5)
        result_changed_order = ap_at_k(range(1, 6), [7, 6, 4, 2, 1], k=5)
        assert pytest.approx(result_ap_at_k) != result_changed_order

    def test_map_k(self):
        result_map_at_k = map_at_k([range(1, 5)], [range(1, 5)], k=3)
        assert pytest.approx(result_map_at_k) == 1.0

        result_map_at_k = map_at_k(
            actual=[[1, 3, 4], [1, 2, 4], [1, 3]],
            predicted=[range(1, 6), range(1, 6), range(1, 6)],
            k=3,
        )
        assert pytest.approx(result_map_at_k, rel=0.001) == 0.685

        result_map_at_k = map_at_k(
            actual=[range(1, 6), range(1, 6)],
            predicted=[[6, 4, 7, 1, 2], [1, 1, 1, 1, 1]],
            k=5,
        )
        assert pytest.approx(result_map_at_k) == 0.26

        result_map_at_k = map_at_k(
            actual=[[1, 3], [1, 2, 3], [1, 2, 3]],
            predicted=[range(1, 6), [1, 1, 1], [1, 2, 1]],
            k=3,
        )
        assert pytest.approx(result_map_at_k) == 11 / 18
