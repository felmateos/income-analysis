from income_kedro.pipelines.income.pipeline import create_pipeline


def register_pipelines():  # pragma: no cover
    return {
        "income": create_pipeline(),
        "__default__": create_pipeline(),
    }
