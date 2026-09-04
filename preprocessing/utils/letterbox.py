"""Redimensionamento com preservacao da proporcao da imagem."""


import cv2
import numpy as np


def letterbox(
    frame: np.ndarray,
    target_size: int = 640,
    pad_color: int = 114,
) -> tuple[np.ndarray, float, tuple[int, int]]:
    """Redimensiona e centraliza ``frame`` em uma tela quadrada."""
    height, width = frame.shape[:2]
    scale = min(target_size / height, target_size / width)
    new_width = round(width * scale)
    new_height = round(height * scale)

    resized = cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
    pad_w = (target_size - new_width) // 2
    pad_h = (target_size - new_height) // 2

    output_shape = (target_size, target_size, *frame.shape[2:])
    output = np.full(output_shape, pad_color, dtype=frame.dtype)
    output[pad_h : pad_h + new_height, pad_w : pad_w + new_width] = resized
    return output, scale, (pad_w, pad_h)


def adjust_bboxes(
    boxes_xyxy: np.ndarray,
    scale: float,
    pad_w: int,
    pad_h: int,
) -> np.ndarray:
    """Mapeia caixas ``xyxy`` do letterbox de volta a imagem original."""
    boxes = boxes_xyxy.copy().astype(float)
    boxes[:, [0, 2]] = (boxes[:, [0, 2]] - pad_w) / scale
    boxes[:, [1, 3]] = (boxes[:, [1, 3]] - pad_h) / scale
    return boxes
