import pytest

from src.gid_ml_framework.pipelines.ranking.nodes import (
    _calculate_map,
    _prepare_groups_for_ranking,
    train_val_split,
)


# _prepare_groups_for_ranking()
def test_prepare_groups_for_ranking(candidates_dummy_df):
    result_np = _prepare_groups_for_ranking(candidates_dummy_df)
    assert len(result_np) == candidates_dummy_df["customer_id"].nunique()


# train_val_split()
class TestTrainValSplit:
    def test_no_downsampling(self, candidates_dummy_df):
        initial_df = candidates_dummy_df.copy()
        num_positives = 40
        val_size = 0.2

        result_train_df, result_val_df = train_val_split(
            initial_df, val_size=val_size, downsampling=False
        )
        assert len(initial_df) == len(result_train_df) + len(result_val_df)
        assert result_val_df["label"].sum() > 0
        assert (
            result_val_df["label"].sum() + result_train_df["label"].sum()
            == num_positives
        )

    def test_no_overlapping_customers(self, candidates_dummy_df):
        initial_df = candidates_dummy_df.copy()
        val_size = 0.1

        result_train_df, result_val_df = train_val_split(
            initial_df, val_size=val_size, downsampling=False
        )

        train_customers = set(result_train_df["customer_id"].unique())
        val_customers = set(result_val_df["customer_id"].unique())
        assert train_customers.intersection(val_customers) == set()

    def test_downsampling(self, candidates_dummy_df):
        initial_df = candidates_dummy_df.copy()
        num_positives = 40
        neg_samples = 100
        val_size = 0.1

        result_train_df, result_val_df = train_val_split(
            initial_df, val_size=val_size, downsampling=True, neg_sample=neg_samples
        )

        # all positives
        assert (
            result_val_df["label"].sum() + result_train_df["label"].sum()
            == num_positives
        )

        # all positives + number of negative samples
        positives_val = len(result_val_df[result_val_df["label"] == 1])
        all_train = len(result_train_df)
        assert positives_val + all_train == num_positives + neg_samples

    def test_downsampling_neg_sample_bigger_than_df(self, candidates_dummy_df):
        initial_df = candidates_dummy_df.copy()
        neg_samples = 999_999
        val_size = 0.1

        with pytest.raises(ValueError):
            train_val_split(
                initial_df, val_size=val_size, downsampling=True, neg_sample=neg_samples
            )


# _calculate_map()
def test_calculate_map(predictions_dummy_df, val_transactions_dummy_df):
    result = _calculate_map(predictions_dummy_df, val_transactions_dummy_df)
    assert result >= 0 and result <= 1


def test_calculate_map_no_overlap(predictions_dummy_df, val_transactions_dummy_df):
    val_transactions_df = val_transactions_dummy_df.copy()
    val_transactions_df["article_id"] = range(len(val_transactions_df))
    val_transactions_df["article_id"] = val_transactions_df["article_id"].astype(str)

    result = _calculate_map(predictions_dummy_df, val_transactions_df)
    assert pytest.approx(result) == 0.0
