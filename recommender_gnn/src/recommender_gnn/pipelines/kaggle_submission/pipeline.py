from typing import Optional

from kedro.pipeline import Pipeline, node, pipeline

from .nodes import generate_submission


def create_pipeline(
    dataset: str,
    model: str,
    users: str,
    test_df: Optional[str] = None,
    comments: Optional[str] = None,
    **kwargs,
) -> Pipeline:
    """Creates pipeline for generating kaggle submission file from saved predictions

    Args:
        dataset (str): dataset name
        model (str): name of gnn model which was used to generate predictions
        users (str): name of kedro dataset for dataframe with users ids subset for which submission should be generated
            (dataframe in format before mapping)
        test_df (str): name of kedro dataset for dataframe with test transactions or customers, used for filtering
            predictions (dataframe in format before mapping)
    """
    namespace_list = [x for x in [dataset, model, comments] if x is not None]
    namespace = "_".join(["kaggle_submission"] + namespace_list)
    gr_namespace = "_".join(["graph_recommendation"] + namespace_list)
    grp_namespace = "_".join(
        ["graph_recommendation_preprocessing"] + namespace_list[::2]
    )

    pipeline_template = pipeline(
        [
            node(
                func=generate_submission,
                inputs=[
                    "predictions",
                    "users",
                    "users_mapping",
                    "items_mapping",
                    "test_df",
                    "params:new_item_column",
                    "params:new_user_column",
                    "params:original_user_column",
                    "params:filter_by_test_users",
                ],
                outputs="submission",
                name=f"generate_{dataset}_submission_node",
            )
        ]
    )

    main_pipeline = pipeline(
        pipe=pipeline_template,
        inputs={
            "predictions": f"{gr_namespace}.predictions",
            "users": users,
            "users_mapping": f"{grp_namespace}.users_mapping",
            "items_mapping": f"{grp_namespace}.items_mapping",
            "test_df": test_df if test_df else users,
        },
        namespace=namespace,
    )

    return main_pipeline
