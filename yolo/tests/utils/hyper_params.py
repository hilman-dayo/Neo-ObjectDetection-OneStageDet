from dataclasses import dataclass
import torch


def hyper_params_static_attrs():
    return [("cuda", True if torch.cuda.is_available() else False)]


def hyper_params_set_attrs():
    ret = [
        ("labels", "labels"),
        ("classes", "len(self.labels)"),
        ("data_root", "data_root_dir"),
        ("model_name", "model_name"),
    ]
    return ret


def hyper_params_train_static_attrs():
    ret = (hyper_params_static_attrs() +
           [
               ("jitter", 0.3),
               ("flip", 0.5),
               ("hue", 0.1),
               ("sat", 1.5),
               ("val", 1.5),
               ("rs_steps", []),
               ("rs_rates", [])
           ]
           )
    return ret

def hyper_params_train_set_attrs():
    # the hyperparameters and how it is mapped to the val original
    ret = (hyper_params_set_attrs() +
           [
               ("nworkers", "nworkers"),
               ("pin_mem", "pin_mem"),
               ("_dataset", "dataset"),
               ("_trainfile", "self.data_root/self._dataset"),
               ("trainfile", "self._trainfile.pkl"),
               ("network_size", "input_shape"),
               ("batch", "batch_size"),
               ("mini_batch", "mini_batch_size"),
               ("max_batches", "max_batches"),
               ("learning_rate", "warmup_lr"),
               ("momentum", "momentum"),
               ("decay", "decay"),
               ("lr_steps", "lr_steps"),
               ("lr_rates", "lr_rates"),
               ("backup", "backup_interval"),
               ("bp_steps", "backup_steps"),
               ("bp_rates", "backup_rates"),
               ("backup_dir", "backup_dir"),
               ("resize", "resize_interval"),
               ("weights", "weights"),
               ("clear", "clear"),
           ])
    return ret


def hyper_params_train_static_attrs():
    ret = (hyper_params_static_attrs() +
           [
               ("jitter", 0.3),
               ("flip", 0.5),
               ("hue", 0.1),
               ("sat", 1.5),
               ("val", 1.5),
               ("rs_steps", []),
               ("rs_rates", [])
           ]
           )
    return ret


def hyper_params_test_static_attrs():
    return hyper_params_static_attrs()
