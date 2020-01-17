"""Test everything related to the hyperparameters."""

import pytest


class TestDefaultTrainYolov3Config():
    """Test Default Yolov3 Test Config.

    Here, we know that the name of variables in config is the same as been
    defined in the *yml file.
    At the end of the test, we will assert weather everything is checked or not.
    """
    def test_output_settings(self, dy3c, dy3kv):
        assert dy3kv.pop("output_root") == dy3c["output_root"]
        assert dy3kv.pop("output_version") == dy3c["output_version"]
        assert dy3kv.pop("backup_name") == dy3c["backup_name"]
        assert dy3kv.pop("log_name") == dy3c["log_name"]

    def test_data_settings(self, dy3c, dy3kv):
        assert dy3kv.pop("data_root_dir") == dy3c["data_root_dir"]

    def test_label_setting(self, dy3c, dy3kv):
        assert dy3kv.pop("labels") == dy3c["labels"]

    def test_train_settings(self, dy3c, dy3kv):
        assert dy3kv.pop("dataset") == dy3c["dataset"]
        assert dy3kv.pop("model_name") == dy3c["model_name"]
        assert dy3kv.pop("stdout") == dy3c["stdout"]
        assert dy3kv.pop("gpus") == dy3c["gpus"]
        assert dy3kv.pop("nworkers") == dy3c["nworkers"]
        assert dy3kv.pop("pin_mem") == dy3c["pin_mem"]
        assert dy3kv.pop("momentum") == dy3c["momentum"]
        assert dy3kv.pop("decay") == dy3c["decay"]
        assert dy3kv.pop("clear") == dy3c["clear"]
        assert dy3kv.pop("lr_steps") == dy3c["lr_steps"]
        assert dy3kv.pop("max_batches") == dy3c["max_batches"]
        assert dy3kv.pop("resize_interval") == dy3c["resize_interval"]
        assert dy3kv.pop("backup_interval") == dy3c["backup_interval"]
        assert dy3kv.pop("backup_steps") == dy3c["backup_steps"]
        assert dy3kv.pop("backup_rates") == dy3c["backup_rates"]
        assert dy3kv.pop("input_shape") == dy3c["input_shape"]
        assert dy3kv.pop("batch_size") == dy3c["batch_size"]
        assert dy3kv.pop("mini_batch_size") == dy3c["mini_batch_size"]
        # Internally, an empty weight is treated as `None`
        if dy3kv["weights"] == "":
            dy3kv["weights"] = None
        assert dy3kv.pop("weights") == dy3c["weights"]
        assert dy3kv.pop("backup_dir") == dy3c["backup_dir"]

    @pytest.mark.xfail
    def test_need_to_be_confirmed_1(self, dy3kv, dy3c):
        # yaml parser may be the problem
        assert dy3kv.pop("warmup_lr") == dy3c["warmup_lr"]

    @pytest.mark.xfail
    def test_need_to_be_confirmed_2(self, dy3kv, dy3c):
        # yaml parser may be the problem
        assert dy3kv.pop("lr_rates") == dy3c["lr_rates"]

    def test_all_tested(self, dy3kv):
        assert len(dy3kv) == 0


class TestDefaultTrainYolov3Hyperparams():
    def test_output_settings(self, dy3h):
        # CONT: test this
        print(dy3h)
