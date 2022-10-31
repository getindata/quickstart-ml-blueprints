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
        "product_code": {0: 709746, 1: 888113, 2: 621372, 3: 759597, 4: 814213},
        "prod_name": {
            0: "Louie Heel",
            1: "Sushi trousers",
            2: "PE Aurora dress 2",
            3: "Nick muscle fit trs",
            4: "Princeton ss shirt (TVP)",
        },
        "product_type_no": {0: 90, 1: 272, 2: 265, 3: 272, 4: 259},
        "product_type_name": {
            0: "Pumps",
            1: "Trousers",
            2: "Dress",
            3: "Trousers",
            4: "Shirt",
        },
        "product_group_name": {
            0: "Shoes",
            1: "Garment Lower body",
            2: "Garment Full body",
            3: "Garment Lower body",
            4: "Garment Upper body",
        },
        "graphical_appearance_no": {
            0: 1010016,
            1: 1010016,
            2: 1010017,
            3: 1010016,
            4: 1010001,
        },
        "graphical_appearance_name": {
            0: "Solid",
            1: "Solid",
            2: "Stripe",
            3: "Solid",
            4: "All over pattern",
        },
        "colour_group_code": {0: 50, 1: 13, 2: 73, 3: 9, 4: 10},
        "colour_group_name": {
            0: "Other Pink",
            1: "Beige",
            2: "Dark Blue",
            3: "Black",
            4: "White",
        },
        "perceived_colour_value_id": {0: 5, 1: 2, 2: 4, 3: 4, 4: 3},
        "perceived_colour_value_name": {
            0: "Bright",
            1: "Medium Dusty",
            2: "Dark",
            3: "Dark",
            4: "Light",
        },
        "perceived_colour_master_id": {0: 4, 1: 11, 2: 2, 3: 5, 4: 9},
        "perceived_colour_master_name": {
            0: "Pink",
            1: "Beige",
            2: "Blue",
            3: "Black",
            4: "White",
        },
        "department_no": {0: 3528, 1: 1941, 2: 3080, 3: 8616, 4: 7657},
        "department_name": {
            0: "Heels",
            1: "Blouse & Dress",
            2: "Take Care External",
            3: "Trouser S&T",
            4: "Kids Boy Shirt",
        },
        "index_code": {0: "C", 1: "A", 2: "A", 3: "F", 4: "H"},
        "index_name": {
            0: "Ladies Accessories",
            1: "Ladieswear",
            2: "Ladieswear",
            3: "Menswear",
            4: "Children Sizes 92-140",
        },
        "index_group_no": {0: 1, 1: 1, 2: 1, 3: 3, 4: 4},
        "index_group_name": {
            0: "Ladieswear",
            1: "Ladieswear",
            2: "Ladieswear",
            3: "Menswear",
            4: "Baby/Children",
        },
        "section_no": {0: 64, 1: 18, 2: 97, 3: 23, 4: 46},
        "section_name": {
            0: "Womens Shoes",
            1: "Womens Trend",
            2: "Collaborations",
            3: "Men Suits & Tailoring",
            4: "Kids Boy",
        },
        "garment_group_no": {0: 1020, 1: 1010, 2: 1001, 3: 1009, 4: 1011},
        "garment_group_name": {
            0: "Shoes",
            1: "Blouses",
            2: "Unknown",
            3: "Trousers",
            4: "Shirts",
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
