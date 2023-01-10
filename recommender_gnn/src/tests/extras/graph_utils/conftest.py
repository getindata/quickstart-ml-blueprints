import pickle

import numpy as np
import pytest


@pytest.fixture
def scores_custom():
    pickle_path = "src/tests/fixtures/arrays/score_array.pkl"
    with open(pickle_path, "rb") as f:
        score = pickle.load(f)
    scores = np.split(score[0], [10, 20, 30, 40])
    return scores
