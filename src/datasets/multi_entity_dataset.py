from src.datasets.dataset import get_events
from src.datasets.entities_names import entities_dict


class MultiEntityDataset:
    def __init__(self, dataset_class, seed: int=42, ds_kwargs: dict={}):
        name = dataset_class(seed=seed, **ds_kwargs).name.split("_")[0]
        if "msl" in name or "smap" in name or "smd" in name:
            name = name.split("-")[0]
        self.dataset_class = dataset_class

        if name in entities_dict.keys():
            self.entity_names = entities_dict[name]
        else:
            self.entity_names = [name]

        self.num_entities = len(self.entity_names)
        self.seed = seed
        self.ds_kwargs = ds_kwargs
        self.name = name + "_me"
        self.datasets = self._datasets()

    def _datasets(self):
        datasets = [self.dataset_class(seed=self.seed, **self.ds_kwargs, entity=entity) for entity in self.entity_names]
        return datasets
