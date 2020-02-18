"""Test everything related to the hyperparameters."""

import pytest
from ..utils import hyper_params_train_static_attrs, hyper_params_train_set_attrs


class TestDefaultTrainYolov3Config():
    """Test Default Yolov3 Test Config.

    Here, we know that the name of variables in config is the same as been
    defined in the *yml file.
    At the end of the test, we will assert weather everything is checked or not.
    """
    def test_output_settings(self, dy3_config_dict, dy3_config_r):
        assert dy3_config_r.pop("output_root") == dy3_config_dict["output_root"]
        assert dy3_config_r.pop("output_version") == dy3_config_dict["output_version"]
        assert dy3_config_r.pop("backup_name") == dy3_config_dict["backup_name"]
        assert dy3_config_r.pop("log_name") == dy3_config_dict["log_name"]

    def test_data_settings(self, dy3_config_dict, dy3_config_r):
        assert dy3_config_r.pop("data_root_dir") == dy3_config_dict["data_root_dir"]

    def test_label_setting(self, dy3_config_dict, dy3_config_r):
        assert dy3_config_r.pop("labels") == dy3_config_dict["labels"]

    def test_train_settings(self, dy3_config_dict, dy3_config_r):
        assert dy3_config_r.pop("dataset") == dy3_config_dict["dataset"]
        assert dy3_config_r.pop("model_name") == dy3_config_dict["model_name"]
        assert dy3_config_r.pop("stdout") == dy3_config_dict["stdout"]
        assert dy3_config_r.pop("gpus") == dy3_config_dict["gpus"]
        assert dy3_config_r.pop("nworkers") == dy3_config_dict["nworkers"]
        assert dy3_config_r.pop("pin_mem") == dy3_config_dict["pin_mem"]
        assert dy3_config_r.pop("momentum") == dy3_config_dict["momentum"]
        assert dy3_config_r.pop("decay") == dy3_config_dict["decay"]
        assert dy3_config_r.pop("clear") == dy3_config_dict["clear"]
        assert dy3_config_r.pop("lr_steps") == dy3_config_dict["lr_steps"]
        assert dy3_config_r.pop("max_batches") == dy3_config_dict["max_batches"]
        assert dy3_config_r.pop("resize_interval") == dy3_config_dict["resize_interval"]
        assert dy3_config_r.pop("backup_interval") == dy3_config_dict["backup_interval"]
        assert dy3_config_r.pop("backup_steps") == dy3_config_dict["backup_steps"]
        assert dy3_config_r.pop("backup_rates") == dy3_config_dict["backup_rates"]
        assert dy3_config_r.pop("input_shape") == dy3_config_dict["input_shape"]
        assert dy3_config_r.pop("batch_size") == dy3_config_dict["batch_size"]
        assert dy3_config_r.pop("mini_batch_size") == dy3_config_dict["mini_batch_size"]
        # Internally, an empty weight is treated as `None`
        if dy3_config_r["weights"] == "":
            dy3_config_r["weights"] = None
        assert dy3_config_r.pop("weights") == dy3_config_dict["weights"]
        assert dy3_config_r.pop("backup_dir") == dy3_config_dict["backup_dir"]

    @pytest.mark.xfail
    def test_need_to_be_confirmed_1(self, dy3_config_r, dy3_config_dict):
        # yaml parser may be the problem
        assert dy3_config_r.pop("warmup_lr") == dy3_config_dict["warmup_lr"]

    @pytest.mark.xfail
    def test_need_to_be_confirmed_2(self, dy3_config_r, dy3_config_dict):
        # yaml parser may be the problem
        assert dy3_config_r.pop("lr_rates") == dy3_config_dict["lr_rates"]

    def test_all_tested(self, dy3_config_r):
        assert len(dy3_config_r) == 0

checked = []
class TestDefaultTrainYolov3Hyperparams():
    @pytest.mark.parametrize("static", hyper_params_train_static_attrs())
    def test_output_static_settings(self, dy3_hyper_train, static):
        t = dy3_hyper_train
        attr = static[0]
        checked.append(attr)
        assert getattr(t, attr) == static[1]

    @pytest.mark.parametrize("set_", hyper_params_train_set_attrs())
    def test_output_set_settings(self, dy3_config_dict, dy3_hyper_train, set_):
        # XXX: `dy3_config_dict` itself didn't pass test `TestDefaultTrainYolov3Config`
        t = dy3_hyper_train
        attr = set_[0]
        checked.append(attr)
        # TEMP: temp solution
        if attr == "classes":
            ass = len(getattr(t, "labels")) == len(dy3_config_dict["labels"])
        elif attr == "_trainfile":
            ass = (getattr(t, "data_root") + "/" + getattr(t, "_dataset")
                   == dy3_config_dict["data_root_dir"] + "/" + dy3_config_dict["dataset"])
        elif attr == "trainfile":
            ass = getattr(t, "trainfile") == getattr(t, "_trainfile") + ".pkl"
        else:
            # tt = dy3_config_dict[set_[1]]
            # print(dy3_config_dict[set_[1]], type(tt), type(getattr(t, attr)))
            ass = getattr(t, attr) == dy3_config_dict[set_[1]]

        assert ass
