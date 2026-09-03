#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path


EXPECTED_CLASSES = ["Capacete", "Colete", "Pessoa"]
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_names(data_yaml: Path) -> list[str]:
    text = data_yaml.read_text(encoding="utf-8")
    for line in text.splitlines():
        if line.strip().startswith("names:"):
            raw = line.split(":", 1)[1].strip()
            return [part.strip().strip("'\"") for part in raw.strip("[]").split(",")]
    raise SystemExit(f"[ERRO] Campo names nao encontrado em {data_yaml}")


def count_split(root: Path, split: str) -> tuple[int, int, int]:
    images_dir = root / split / "images"
    labels_dir = root / split / "labels"
    if not images_dir.is_dir() or not labels_dir.is_dir():
        raise SystemExit(f"[ERRO] Split {split} sem pastas images/labels")

    images = sorted(p for p in images_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS)
    labels = sorted(labels_dir.glob("*.txt"))
    label_stems = {p.stem for p in labels}
    missing_labels = [p.name for p in images if p.stem not in label_stems]
    empty_labels = [p.name for p in labels if not p.read_text(encoding="utf-8").strip()]

    if missing_labels:
        raise SystemExit(f"[ERRO] {split}: {len(missing_labels)} imagens sem label")
    if empty_labels:
        raise SystemExit(f"[ERRO] {split}: {len(empty_labels)} labels vazios")

    return len(images), len(labels), len(empty_labels)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--min-train", type=int, default=350)
    args = parser.parse_args()

    root = args.dataset.resolve()
    names = parse_names(root / "data.yaml")
    if names != EXPECTED_CLASSES:
        raise SystemExit(f"[ERRO] Classes esperadas {EXPECTED_CLASSES}, encontradas {names}")

    counts = {split: count_split(root, split) for split in ("train", "valid", "test")}
    train_images = counts["train"][0]
    total_images = sum(value[0] for value in counts.values())

    if train_images < args.min_train:
        raise SystemExit(f"[ERRO] Treino com {train_images} imagens, minimo {args.min_train}")

    print("[OK] Dataset aprovado")
    print(f"[OK] Classes: {', '.join(names)}")
    print(f"[OK] Train: {counts['train'][0]} imagens / {counts['train'][1]} labels")
    print(f"[OK] Valid: {counts['valid'][0]} imagens / {counts['valid'][1]} labels")
    print(f"[OK] Test: {counts['test'][0]} imagens / {counts['test'][1]} labels")
    print(f"[OK] Total: {total_images} imagens anotadas")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
