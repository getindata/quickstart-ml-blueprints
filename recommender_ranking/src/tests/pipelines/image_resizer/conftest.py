import numpy as np
import pytest
from PIL import Image


@pytest.fixture
def dummy_img():
    img = np.asarray(
        Image.open("src/tests/fixtures/img/from/0550827007.jpg").convert("RGB")
    )
    return img
