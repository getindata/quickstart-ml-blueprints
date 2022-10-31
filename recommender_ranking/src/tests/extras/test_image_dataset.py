import numpy as np

from src.gid_ml_framework.extras.datasets.image_dataset import (
    DirWithImagesDataSet,
)


class TestDirWithImagesDataSet:
    # loading
    def test_load_returns_dict(self):
        dir = "src/tests/fixtures/img/from/"
        load_args = {"img_extension": "jpg"}
        dir_with_images = DirWithImagesDataSet(dir=dir, load_args=load_args)
        img_dict = dir_with_images._load()
        assert isinstance(img_dict, dict)
        assert len(img_dict) == 1

    def test_load_img_name_as_key(self):
        dir = "src/tests/fixtures/img/from/"
        load_args = {"img_extension": "jpg"}
        dir_with_images = DirWithImagesDataSet(dir=dir, load_args=load_args)
        img_dict = dir_with_images._load()
        assert list(img_dict.keys())[0] == "0550827007"

    def test_load_ndarray_as_value(self):
        dir = "src/tests/fixtures/img/from/"
        load_args = {"img_extension": "jpg"}
        dir_with_images = DirWithImagesDataSet(dir=dir, load_args=load_args)
        img_dict = dir_with_images._load()
        assert isinstance(img_dict.get("0550827007"), np.ndarray)

    # saving
    def test_save(self):
        load_dir = "src/tests/fixtures/img/from/"
        load_args = {"img_extension": "jpg"}
        load_dir_with_images = DirWithImagesDataSet(dir=load_dir, load_args=load_args)
        img_dict = load_dir_with_images._load()

        save_dir = "src/tests/fixtures/img/to/"
        save_args = {"img_extension": "jpg"}
        save_dir_with_images = DirWithImagesDataSet(dir=save_dir, save_args=save_args)
        save_dir_with_images._save(data=img_dict)
