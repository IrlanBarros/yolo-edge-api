"""Testes automatizados do pipeline reutilizavel de pre-processamento."""

import numpy as np

from preprocessing.preprocessor import (
    CONFIG_DEFAULT,
    CONFIG_HIGH_QUALITY,
    CONFIG_LOW_LIGHT,
    PreprocessConfig,
    Preprocessor,
)


def make_frame(height=480, width=640):
    rng = np.random.default_rng(42)
    return rng.integers(0, 256, (height, width, 3), dtype=np.uint8)


def test_output_shape_letterbox():
    result = Preprocessor(PreprocessConfig(infer_size=416)).process(make_frame())
    assert result.frame.shape == (416, 416, 3)


def test_output_dtype_uint8_without_normalization():
    result = Preprocessor(PreprocessConfig(normalize=False)).process(make_frame())
    assert result.frame.dtype == np.uint8


def test_output_is_float32_and_bounded_when_normalized():
    result = Preprocessor(PreprocessConfig(normalize=True)).process(make_frame())
    assert result.frame.dtype == np.float32
    assert 0.0 <= result.frame.min() <= result.frame.max() <= 1.0


def test_letterbox_metadata_and_symmetric_padding():
    result = Preprocessor(PreprocessConfig(infer_size=416)).process(make_frame())
    assert result.scale == 0.65
    assert result.orig_size == (480, 640)
    assert result.pad_w == 0
    assert result.pad_h == 52


def test_adjust_boxes_removes_letterbox_transform():
    processor = Preprocessor(PreprocessConfig(infer_size=416))
    result = processor.process(make_frame())
    original = np.array([[100.0, 80.0, 500.0, 400.0]])
    letterboxed = original.copy()
    letterboxed[:, [0, 2]] = letterboxed[:, [0, 2]] * result.scale_x + result.pad_w
    letterboxed[:, [1, 3]] = letterboxed[:, [1, 3]] * result.scale_y + result.pad_h
    np.testing.assert_allclose(processor.adjust_boxes(letterboxed, result), original)


def test_adjust_boxes_without_letterbox_uses_both_axis_scales():
    processor = Preprocessor(PreprocessConfig(infer_size=416, use_letterbox=False))
    result = processor.process(make_frame())
    boxes = processor.adjust_boxes(np.array([[0, 0, 416, 416]]), result)
    np.testing.assert_allclose(boxes, [[0, 0, 640, 480]])
    assert result.scale_x != result.scale_y


def test_low_light_config_applies_clahe_and_keeps_three_channels():
    processor = Preprocessor(CONFIG_LOW_LIGHT)
    result = processor.process(make_frame())
    assert processor.cfg.clahe
    assert result.frame.shape == (320, 320, 3)


def test_predefined_default_and_high_quality_configs():
    assert not CONFIG_DEFAULT.gaussian_blur
    assert not CONFIG_DEFAULT.median_blur
    assert not CONFIG_DEFAULT.clahe
    result = Preprocessor(CONFIG_HIGH_QUALITY).process(make_frame())
    assert result.frame.shape == (640, 640, 3)

