import pandas as pd

from src.gid_ml_framework.pipelines.feature_selection_apply.nodes import (
    apply_feature_selection,
)


def test_all_columns_selected(articles_dummy_df):
    initial_df = articles_dummy_df
    columns_selected = set(
        [
            "article_id",
            "product_code",
            "prod_name",
            "product_type_no",
            "product_type_name",
            "product_group_name",
            "graphical_appearance_no",
            "graphical_appearance_name",
            "colour_group_code",
            "colour_group_name",
            "perceived_colour_value_id",
            "perceived_colour_value_name",
            "perceived_colour_master_id",
            "perceived_colour_master_name",
            "department_no",
            "department_name",
            "index_code",
            "index_name",
            "index_group_no",
            "index_group_name",
            "section_no",
            "section_name",
            "garment_group_no",
            "garment_group_name",
            "detail_desc",
        ]
    )
    result_df = apply_feature_selection(articles_dummy_df, columns_selected)
    pd.testing.assert_frame_equal(result_df, initial_df)


def test_few_correct_columns_selected(articles_dummy_df):
    few_columns_selected = set(
        [
            "article_id",
            "product_code",
            "product_type_no",
            "product_type_name",
            "product_group_name",
            "graphical_appearance_no",
            "department_no",
            "department_name",
            "index_code",
            "index_name",
            "garment_group_no",
            "garment_group_name",
            "detail_desc",
        ]
    )
    result_df = apply_feature_selection(articles_dummy_df, few_columns_selected)
    assert len(result_df.columns) == len(few_columns_selected)


def test_columns_missing_in_the_dataset_selected(articles_dummy_df):
    few_columns_selected = set(
        [
            "article_id",
            "product_code",
            "product_type_no",
            "product_type_name",
            "product_group_name",
            "graphical_appearance_no",
            "department_no",
            "department_name",
            "index_code",
            "index_name",
            "garment_group_no",
            "garment_group_name",
            "detail_desc",
        ]
    )
    not_existing_columns = set(["does_not_exist_1", "does_not_exist2"])
    selected_columns = few_columns_selected.union(not_existing_columns)
    result_df = apply_feature_selection(articles_dummy_df, selected_columns)
    assert len(result_df.columns) == len(few_columns_selected)


def test_empty_set_selected(articles_dummy_df):
    empty_set = set()
    result_df = apply_feature_selection(articles_dummy_df, empty_set)
    assert result_df.empty is True
