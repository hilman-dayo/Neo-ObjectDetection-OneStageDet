import pytest
from vedanet.engine.engine import Engine
from utils import envs
import yaml
from pathlib import Path
import vedanet as vn

from .utils import AnnoPickleDatasetFactory


@pytest.fixture(scope="session")
def yolo_dir(tmp_path_factory):
    yolo_d = tmp_path_factory.mktemp("yolo")
    pass


@pytest.fixture(scope='module')
def monkeymodule():
    """Monkey patch for a module session.

    Too bad the default one is function scoped.
    """
    # https://stackoverflow.com/questions/53963822/python-monkeypatch-setattr-with-pytest-fixture-at-module-scope
    from _pytest.monkeypatch import MonkeyPatch
    mpatch = MonkeyPatch()
    yield mpatch
    mpatch.undo()


# TODO: Make a global `factory` one (test, train, speed)
@pytest.fixture(name="dy3_config_r", scope="class")
def default_yolov3_config_keyval_setting_mapping():
    """Mapping of settings name in yml file with the one loaded to framework.

    For some reasons, the settings defined in yml file are not the same when
    it is loaded to the framework. In this fixture, we defined a dict with
    name of settings when loaded to framework as the key and its value.
    Then, we use it to prepare a mock version of setting file
    (refer `default_yolov3_config`).

    From here, we know two things:
    1. How the settings name mapping works.
    2. What kind of settings are being passed.
    3. What will be the type be when it is passed.
    """
    # For this time being, this is only for training.
    model_name = "Yolov3"
    return {
        "output_root": "outputs",
        "output_version": "baseline",
        "backup_name": "weights",
        "log_name": "logs",
        "labels": ["aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car",
                   "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike",
                   "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"],
        "data_root_dir": "/dayo/datasets/voc_onestagedet/VOCdevkit/onedet_cache",
        # This is one is set at CL
        "model_name": model_name,
        "dataset": "train",
        "stdout": True,
        "gpus": "0",
        "nworkers": 16,
        "pin_mem": True,
        "momentum": 0.9,
        "decay": 0.0005,
        "clear": False,
        "warmup_lr": 0.00005,
        "lr_steps": [400,700,900,1000, 40000,45000],
        "lr_rates": [0.0001,0.0002,0.0005,0.001, 0.0001,0.00001],
        "max_batches": 50200,
        "resize_interval": 10,
        "backup_interval": 200,
        "backup_steps": [1000],
        "backup_rates": [10000],
        "input_shape": [608, 608],
        "batch_size": 64,
        "mini_batch_size": 8,
        "weights": "",  # This is to train from existing weights
        # The only settings that is hardcoded...
        # TODO: Prevent backup_dir from being created here...
        "backup_dir": str(Path("outputs") / model_name / "baseline" / "weights"),
        }


@pytest.fixture(name="dy3_config_dict", scope="class")
def default_yolov3_config(monkeymodule, dy3_config_r):
    """Default config made from yolov3 yml file.

    The yml file in cfgs/*.yml defines four things.
    1. Environment setting
    2. Training setting
    3. Test setting
    4. Speed testing setting

    This fixture gives the default one, which is the Yolov3 settings
    that we can work with.
    """
    # SET #####################################################################
    # We set the default settings as a string.
    # Then, we monkey patch `initEnv` to get pass a function that kinda
    # hard coded.
    # TODO: Rename initEnv to init_env and make it better...
    # TODO: Solve the loading yml warning.
    s = dy3_config_r
    yolov3 = f'''
        output_root: "{s['output_root']}"
        output_version: "{s['output_version']}"
        backup_name: "{s['backup_name']}"
        log_name: "{s['log_name']}"

        labels: {s['labels']}

        data_root_dir: "{s['data_root_dir']}"

        train:
            dataset: "{s['dataset']}"
            stdout: {s['stdout']}
            gpus: "{s['gpus']}"
            nworkers: {s['nworkers']}
            pin_mem: {s['pin_mem']}

            momentum: {s['momentum']}
            decay: {s['decay']}

            clear: {s['clear']}

            warmup_lr: {s['warmup_lr']}
            lr_steps: {s['lr_steps']}
            lr_rates: {s['lr_rates']}
            max_batches: {s['max_batches']}
            resize_interval: {s['resize_interval']}

            backup_interval: {s['backup_interval']}
            backup_steps: {s['backup_steps']}
            backup_rates: {s['backup_rates']}

            input_shape: {s['input_shape']}
            batch_size: {s['batch_size']}
            mini_batch_size: {s['mini_batch_size']}
            weights: {s['weights']}

        test:
            dataset: "test"
            stdout: True
            gpus: "3"
            nworkers: 8
            pin_mem: True

            input_shape: [544, 544]
            batch_size: 8
            weights: "weights/yolov3_50200.dw"

            conf_thresh: 0.005
            nms_thresh: 0.45

            results: "results"

        speed:
            gpus: "7"
            batch_size: 1
            max_iters: 200
            input_shape: [544, 544]
    '''
    monkeymodule.setattr(
        envs, "getConfig",
        lambda *args, **kwargs: yaml.load(yolov3)
    )
    train_flag = 1  # Means train
    config = envs.initEnv(train_flag, "Yolov3")
    return config

@pytest.fixture(name="dy3_hyper_train", scope="class")
def default_yolov3_hyperparams_train(dy3_config_dict):
    """Default YOLOv3 hyperparameters made from the config."""
    # XXX: `dy3_config_dict` itself is not perfect
    return vn.hyperparams.HyperParams(dy3_config_dict, train_flag=1)


@pytest.fixture(name="d_vocd_t", scope="class")
def default_voc_dataset_train(dy3_hyper_train):
    """Default YOLOv3 hyperparameters made from the config."""
    return AnnoPickleDatasetFactory.get_dataset("default_voc")(dy3_hyper_train)


@pytest.fixture(scope="function")
def default_env():
    """Produce default environment.

    The *yml file defines
    """
    pass

@pytest.fixture(scope="class")
def default_empty_yolov3_engine(dy3_config_r):
    """Get the "empty" `Engine` instance.

    This engine will contain made-up function. It shows what kind of things you
    need to implement at minimum.
    """
    class EmptyEngine(Engine, hyper_params):
        def __init__(self):
            # super()
            pass  # Need to implement hyper parameter first

    engine = Engine()
    return engine
