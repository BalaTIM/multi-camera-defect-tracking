"""
Frame Preprocessing
--------------------
Resizes and normalizes frames before YOLO inference.
Handles aspect-ratio-preserving letterboxing.
"""

import cv2
import numpy as np


def letterbox(
    img: np.ndarray,
    new_shape: tuple[int, int] = (640, 640),
    color: tuple[int, int, int] = (114, 114, 114),
    auto: bool = True,
    stride: int = 32,
) -> tuple[np.ndarray, tuple[float, float], tuple[int, int]]:
    """
    Resize image to new_shape with letterboxing (preserves aspect ratio).

    Returns:
        img        : resized + padded image
        ratio      : (width_ratio, height_ratio)
        pad        : (dw, dh) padding applied
    """
    shape = img.shape[:2]  # h, w
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    new_unpad = (int(round(shape[1] * r)), int(round(shape[0] * r)))
    dw = new_shape[1] - new_unpad[0]
    dh = new_shape[0] - new_unpad[1]

    if auto:
        dw = dw % stride
        dh = dh % stride

    dw /= 2
    dh /= 2

    if shape[::-1] != new_unpad:
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)

    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)

    return img, (r, r), (dw, dh)


def bgr_to_rgb_tensor(img: np.ndarray) -> np.ndarray:
    """Convert BGR uint8 HxWx3 to float32 RGB normalized [0,1]."""
    img = img[:, :, ::-1]  # BGR -> RGB
    img = img.astype(np.float32) / 255.0
    return img


def enhance_contrast(img: np.ndarray) -> np.ndarray:
    """
    CLAHE contrast enhancement — useful for dark/matte industrial surfaces.
    Operates in LAB color space.
    """
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    merged = cv2.merge((cl, a, b))
    return cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)
