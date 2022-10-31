import pandas as pd
import pytest


@pytest.fixture
def articles_dummy_df():
    articles_dict = {
        "article_id": {
            0: "0709746007",
            1: "0888113001",
            2: "0621372001",
            3: "0759597001",
            4: "0814213002",
        },
        "detail_desc": {
            0: "Sandals with covered block heels and imitation leather insoles. Heel 8 cm.",
            1: "Oversized trousers in twill made from a cotton. High waist with pleats, side pockets.",
            2: "Calf-length, straight-cut wrap dress. Small stand-up collar, long wide sleeves.",
            3: "Suit trousers with a hook-and-eye fastener and zip fly. Side pockets, muscle fit.",
            4: "Short-sleeved shirt with a collar, yoke at the back, and a gently rounded hem.",
        },
    }
    df = pd.DataFrame(articles_dict)
    return df
