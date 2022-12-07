from kedro.pipeline import Pipeline, node, pipeline

from .nodes import generate_submission


def create_pipeline(
    dataset: str, model: str, users: str, test_df, comments: str = None, **kwargs
) -> Pipeline:
    """Creates pipeline for generating kaggle submission file from saved predictions

    Args:
        dataset (str): dataset name
        model (str): name of gnn model which was used to generate predictions
        users (str): dataframe with users ids subset for which submission should be generated (before mapping)
        test_df (str): dataframe with test transactions, used for filtering predictions (before mapping)
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
                    "all_users",
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
            "all_users": users,
            "users_mapping": f"{grp_namespace}.users_mapping",
            "items_mapping": f"{grp_namespace}.items_mapping",
            "test_df": test_df,
        },
        namespace=namespace,
    )

    return main_pipeline
