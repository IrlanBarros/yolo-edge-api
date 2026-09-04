"""Pipeline reutilizavel de pre-processamento para inferencia YOLO."""

from dataclasses import dataclass

import cv2
import numpy as np

from preprocessing.utils.letterbox import letterbox


@dataclass(frozen=True)
class PreprocessConfig:
    """Configuracao imutavel aplicada a todos os frames do processador."""

    infer_size: int = 320
    convert_rgb: bool = True
    use_letterbox: bool = True
    gaussian_blur: bool = False
    gaussian_ksize: int = 3
    gaussian_sigma: float = 0.8
    median_blur: bool = False
    median_ksize: int = 3
    clahe: bool = False
    clahe_clip: float = 2.0
    clahe_tile: int = 8
    clahe_space: str = "lab"
    normalize: bool = False


@dataclass(frozen=True)
class PreprocessResult:
    """Imagem processada e metadados necessarios ao pos-processamento."""

    frame: np.ndarray
    scale: float = 1.0
    scale_x: float = 1.0
    scale_y: float = 1.0
    pad_w: int = 0
    pad_h: int = 0
    orig_size: tuple[int, int] = (0, 0)


class Preprocessor:
    """Executa um pipeline deterministico e reutilizavel por frame."""

    def __init__(self, config: PreprocessConfig | None = None):
        self.cfg = config or PreprocessConfig()
        self._validate_config()
        self._clahe = None
        if self.cfg.clahe:
            self._clahe = cv2.createCLAHE(
                clipLimit=self.cfg.clahe_clip,
                tileGridSize=(self.cfg.clahe_tile, self.cfg.clahe_tile),
            )

    def process(self, frame: np.ndarray) -> PreprocessResult:
        """Processa um frame BGR e devolve imagem e transformacao geometrica."""
        if not isinstance(frame, np.ndarray) or frame.ndim != 3 or frame.shape[2] != 3:
            raise ValueError("frame deve ser um array NumPy HxWx3")
        orig_h, orig_w = frame.shape[:2]
        if orig_h == 0 or orig_w == 0:
            raise ValueError("frame nao pode ter dimensoes vazias")

        output = frame.copy()
        if self.cfg.clahe:
            output = self._apply_clahe(output)
        if self.cfg.convert_rgb:
            output = cv2.cvtColor(output, cv2.COLOR_BGR2RGB)
        if self.cfg.gaussian_blur:
            output = cv2.GaussianBlur(
                output,
                (self.cfg.gaussian_ksize, self.cfg.gaussian_ksize),
                sigmaX=self.cfg.gaussian_sigma,
            )
        elif self.cfg.median_blur:
            output = cv2.medianBlur(output, self.cfg.median_ksize)

        if self.cfg.use_letterbox:
            output, scale, (pad_w, pad_h) = letterbox(output, self.cfg.infer_size)
            scale_x = scale_y = scale
        else:
            output = cv2.resize(output, (self.cfg.infer_size, self.cfg.infer_size))
            scale_x = self.cfg.infer_size / orig_w
            scale_y = self.cfg.infer_size / orig_h
            scale = min(scale_x, scale_y)
            pad_w = pad_h = 0

        if self.cfg.normalize:
            output = output.astype(np.float32) / 255.0

        return PreprocessResult(
            frame=output,
            scale=scale,
            scale_x=scale_x,
            scale_y=scale_y,
            pad_w=pad_w,
            pad_h=pad_h,
            orig_size=(orig_h, orig_w),
        )

    def adjust_boxes(self, boxes_xyxy: np.ndarray, result: PreprocessResult) -> np.ndarray:
        """Converte caixas do frame processado para as coordenadas originais."""
        boxes = np.asarray(boxes_xyxy).copy().astype(float)
        if boxes.ndim != 2 or boxes.shape[1] != 4:
            raise ValueError("boxes_xyxy deve ter shape (N, 4)")
        boxes[:, [0, 2]] = (boxes[:, [0, 2]] - result.pad_w) / result.scale_x
        boxes[:, [1, 3]] = (boxes[:, [1, 3]] - result.pad_h) / result.scale_y

        orig_h, orig_w = result.orig_size
        boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, orig_w)
        boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, orig_h)
        return boxes

    def _apply_clahe(self, frame_bgr: np.ndarray) -> np.ndarray:
        if self.cfg.clahe_space == "lab":
            converted = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2LAB)
            first, second, third = cv2.split(converted)
            merged = cv2.merge((self._clahe.apply(first), second, third))
            return cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)

        converted = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
        first, second, third = cv2.split(converted)
        merged = cv2.merge((first, second, self._clahe.apply(third)))
        return cv2.cvtColor(merged, cv2.COLOR_HSV2BGR)

    def _validate_config(self) -> None:
        if self.cfg.infer_size <= 0:
            raise ValueError("infer_size deve ser positivo")
        if self.cfg.gaussian_blur and self.cfg.median_blur:
            raise ValueError("gaussian_blur e median_blur sao mutuamente exclusivos")
        if self.cfg.clahe_space not in {"lab", "hsv"}:
            raise ValueError("clahe_space deve ser 'lab' ou 'hsv'")
        for name, value in (
            ("gaussian_ksize", self.cfg.gaussian_ksize),
            ("median_ksize", self.cfg.median_ksize),
        ):
            if value <= 0 or value % 2 == 0:
                raise ValueError(f"{name} deve ser impar e positivo")


CONFIG_DEFAULT = PreprocessConfig(
    infer_size=320,
    convert_rgb=True,
    use_letterbox=True,
)

CONFIG_LOW_LIGHT = PreprocessConfig(
    infer_size=320,
    convert_rgb=True,
    use_letterbox=True,
    clahe=True,
    clahe_clip=2.0,
    clahe_tile=8,
    clahe_space="lab",
)

CONFIG_HIGH_QUALITY = PreprocessConfig(
    infer_size=640,
    convert_rgb=True,
    use_letterbox=True,
)

