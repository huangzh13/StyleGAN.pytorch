from PIL import Image
import pytest

from config import cfg as default_cfg
from data import make_dataset


def _make_image_folder(root):
    for class_name, color in (("class_a", (255, 0, 0)), ("class_b", (0, 255, 0))):
        class_dir = root / class_name
        class_dir.mkdir()
        Image.new("RGB", (8, 8), color).save(class_dir / "sample.png")


def _dataset_cfg(root):
    cfg = default_cfg.clone()
    cfg.dataset.img_dir = str(root)
    cfg.dataset.folder = False
    cfg.dataset.resolution = 8
    return cfg.dataset


def test_conditional_dataset_returns_imagefolder_labels(tmp_path):
    _make_image_folder(tmp_path)

    dataset = make_dataset(_dataset_cfg(tmp_path), conditional=True, n_classes=2)

    image, label = dataset[0]
    assert dataset.classes == ["class_a", "class_b"]
    assert image.shape == (3, 8, 8)
    assert label in (0, 1)


def test_conditional_dataset_rejects_wrong_class_count(tmp_path):
    _make_image_folder(tmp_path)

    with pytest.raises(ValueError, match="contains 2 classes, but n_classes is 3"):
        make_dataset(_dataset_cfg(tmp_path), conditional=True, n_classes=3)
