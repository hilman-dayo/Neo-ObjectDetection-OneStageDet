import pytest
from brambox.boxes.annotations.pickle import PickleAnnotation
from vedanet.engine._voc_train import VOCDataset
import torch


# TEMP: Make this part of `DefaultVOCDataset`
def voc_default_attr():
    ret = {
        "class_label": str,
        "object_id": int,
        "x_top_left": int,
        "y_top_left": int,
        "width": int,
        "height": int,
        "ignore": bool,
        "lost": None,  # not really sure this
        "difficult": bool,
        "truncated_fraction": float,
        "occluded_fraction": float,
        "visible_x_top_left": float,
        "visible_y_top_left": float,
        "visible_width": float,
        "visible_height": float,
    }

    return ret.items()


class TestDefaultVOCTrainDataset():
    """Understands the behavior of default VOC dataset."""
    def test_class(self, d_vocd_t):
        print(type(d_vocd_t))
        assert isinstance(d_vocd_t, VOCDataset)

    @pytest.mark.parametrize("attrs_type", voc_default_attr())
    @pytest.mark.parametrize("index", list(range(10)))
    def test_attrs(self, d_vocd_t, index, attrs_type):
        image, info = d_vocd_t[index]

        image_t = isinstance(image, torch.Tensor)
        info_t = []
        info_attr_t = []
        for i in info:  # info is a list
            info_t.append(isinstance(i, PickleAnnotation))
            if attrs_type[0] == "lost": # TEMP
                attr_t = True
            else:
                # print(type(i))
                # print([attrs_type[0]])
                attr_t = isinstance(getattr(i, attrs_type[0]), attrs_type[1])
            info_attr_t.append(attr_t)

        assert all([
            image_t,
            *info_t,
            *info_attr_t,
        ])
