import numpy as np
import cv2
from io import BytesIO


def load_image(path: str) -> np.array:
    """
    Load image

    :param path: Path to the image
    :type path: str
    :return: Numpy array representation of the image
    :rtype: np.array
    """
    with path.open("rb") as reader:
        data = np.fromstring(reader.read(), dtype=np.uint8)
        img = cv2.imdecode(data, cv2.IMREAD_COLOR)
        if img is None:
            return
        img = img[..., ::-1]

    return img

def resize_by_max(
    image: np.array, max_side: int = 512, force: bool = False
) -> np.array:
    """
    Function to resize the input image

    :param image: Input image to resize
    :type image: np.array
    :param max_side: New size of the image, defaults to 512
    :type max_side: int, optional
    :param force: Whether to force or not the resize, defaults to False
    :type force: bool, optional
    :return: THe resize image
    :rtype: np.array
    """
    h, w = image.shape[:2]
    if max(h, w) < max_side and not force:
        return image
    ratio = max(h, w) / max_side

    w = int(w / ratio + 0.5)
    h = int(h / ratio + 0.5)
    return cv2.resize(image, (w, h))

def image2buffer(image):
    is_success, buffer = cv2.imencode(".jpg", image)
    if not is_success:
        return None
    return BytesIO(buffer)
