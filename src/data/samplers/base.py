from torch.utils.data.sampler import Sampler
import numpy as np

class BaseSampler(Sampler):

    def __init__(self, dataset, model=None, **kwargs):
        self.dataset = dataset
        self.model = model

        self.targets = dataset.targets
        self.num_classes = len(np.unique(self.targets))

    def _get_indices(self):
        raise NotImplementedError()

    def __len__(self):
        raise NotImplementedError()

    def __iter__(self):
        indices = list(self._get_indices())
        return iter(indices)
