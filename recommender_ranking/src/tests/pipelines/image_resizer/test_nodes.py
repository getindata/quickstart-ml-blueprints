import numpy as np

from src.gid_ml_framework.pipelines.image_resizer.nodes import _resize_image


def test_assert_third_channel(dummy_img):
    initial_third_channel = dummy_img.shape[2]
    resized_img = _resize_image(dummy_img, size=[128, 128], method="bilinear")
    np_resized_img = np.asarray(resized_img)
    transformed_third_channel = np_resized_img.shape[2]
    assert initial_third_channel == transformed_third_channel


class TestResamplingMethods:
    def test_resampling(self, dummy_img):
        resized_img = _resize_image(dummy_img, size=[128, 128], method="bilinear")
        np_resized_img = np.asarray(resized_img)
        assert np_resized_img.shape[0] == 128
        assert np_resized_img.shape[0] == 128

    def test_wrong_resampling(self, dummy_img):
        # if not specified, the default is bicubic
        wrong_method = "does_not_exist"
        wrong_resized_img = _resize_image(
            dummy_img, size=[128, 128], method=wrong_method
        )
        np_wrong_resized_img = np.asarray(wrong_resized_img)
        resized_img = _resize_image(dummy_img, size=[128, 128], method="bicubic")
        np_resized_img = np.asarray(resized_img)
        np.testing.assert_equal(np_resized_img, np_wrong_resized_img)
