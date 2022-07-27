"""
This package contains custom Kedro datasets implementations.

Implementation was based on these sources:
https://kedro.readthedocs.io/en/stable/extend_kedro/custom_datasets.html
https://gitlab.com/getindata/ml-ops/mlops-pytorch-demo/-/blob/master/src/mlops_image_classification/datasets.py
"""
import logging
from typing import Any, Dict, List
from pathlib import Path
from kedro.io import AbstractDataSet
from kedro.io.core import get_protocol_and_path
import numpy as np
from PIL import Image


logger = logging.getLogger(__name__)
Label = str
ImageArray = np.ndarray
ImageMapping = Dict[Label, ImageArray]

class DirWithImagesDataSet(AbstractDataSet):
    """``DirImageDataSet`` loads/saves image data from/to a given directory using Pillow.
    Input: directory with .png images; labels as filenames -> '0321312321.png'
    Reads: dictionary with label as a key, and `np.ndarray` representing image as a value.

    Examples:
    ::
        >>> DirWithImagesDataSet(dir='path/to/images/')
    """
    def __init__(self,
                dir: str,
                save_args: Dict[str, Any] = None, 
                load_args: Dict[str, Any] = None):
        """Creates a new instance of DirWithImagesDataSet to load / save image data for given directory.

        Args:
            dir: The directory with images to load/save data.
        """
        self._dir = Path(dir)
        protocol, _ = get_protocol_and_path(dir)
        self._protocol = protocol
        
        self._load_args = dict()
        if load_args is not None:
            self._load_args.update(load_args)
        
        self._save_args = dict()
        if save_args is not None:
            self._save_args.update(save_args)

    def _load(self) -> ImageMapping:
        img_extension = self._load_args.get('img_extension')
        load_path = self._dir

        img_dict = dict()
        image_names = self._get_file_names(load_path, img_extension)
        for image_name in image_names:
            img_path = load_path / image_name
            img = Image.open(img_path).convert('RGB')
            img_dict[img_path.stem] = np.asarray(img)
        return img_dict

    def _save(self, data: ImageMapping) -> None:
        img_extension = self._save_args.get('img_extension')

        save_path = self._dir
        if save_path.exists():
            logger.warning(f'Directory already exists, it may be not empty!')
        else:
            logger.info(f'Creating new directory: {save_path}')
            save_path.mkdir(parents=False, exist_ok=False)
        
        for label, image in data.items():
            pillow_img = Image.fromarray(image)
            save_filepath = save_path / f'{label}.{img_extension}'
            pillow_img.save(save_filepath)

    def _describe(self) -> Dict[str, Any]:
        """Returns a dict that describes the attributes of the dataset."""
        return dict(filepath=self._dir, protocol=self._protocol)

    def _get_file_names(self, dir: Path, file_extension: str) -> List[str]:
        """Returns a list of file names in a given directory with a specific extension.

        Args:
            dir (Path): directory
            file_extension (str): extension of a file

        Returns:
            List[str]: list of file names
        """
        file_names = [file_name.name for file_name in dir.glob(f"*.{file_extension}")]
        return file_names
