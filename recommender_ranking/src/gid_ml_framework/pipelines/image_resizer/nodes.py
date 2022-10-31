import logging
from typing import Dict, List

import numpy as np
from PIL import Image
from tqdm import tqdm

logger = logging.getLogger(__name__)
Label = str
ImageArray = np.ndarray
ImageMapping = Dict[Label, ImageArray]

resampling_methods = {
    "nearest": Image.Resampling.NEAREST,
    "box": Image.Resampling.BOX,
    "bilinear": Image.Resampling.BILINEAR,
    "hamming": Image.Resampling.HAMMING,
    "bicubic": Image.Resampling.BICUBIC,
    "lanczos": Image.Resampling.LANCZOS,
}


def process_images(
    data: ImageMapping, output_size: List[int] = [128, 128], method: str = "bilinear"
) -> ImageMapping:
    logger.info(f"There are {len(data)} processed images")
    for label, image in tqdm(data.items()):
        resized_image = _resize_image(image, output_size, method)
        data[label] = np.asarray(resized_image)
    return data


def _resize_image(img: np.ndarray, size: List[int], method: str) -> np.ndarray:
    pillow_image = Image.fromarray(img)
    resample = resampling_methods.get(method)
    resized_image = pillow_image.resize(size, resample=resample)
    return resized_image
