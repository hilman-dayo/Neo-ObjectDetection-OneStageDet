from dataclasses import dataclass, field
from vedanet.engine._voc_train import VOCDataset


class AnnoPickleDatasetFactory:
    @staticmethod
    def get_dataset(dataset: str):
        if dataset == "default_voc":
            return VOCDataset


@dataclass
class AnnoPickleDataset:
    pass


# TEMP: This is just to test the already built stuff.
#       Need to make from scratch. Then pass it through AnnoPickleDatasetFactory
@dataclass
class DefaultVOCDataset(AnnoPickleDataset):
    pass
