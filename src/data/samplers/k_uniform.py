import numpy as np
from functools import partial

from .base import BaseSampler

class KUniform(BaseSampler):

    def __init__(self, dataset, k=1000, **kwargs):
        super().__init__(dataset, **kwargs)
        self.k = k
        self.indices = self._chose_indices()

    def __len__(self):
        return self.num_classes * self.k

    def _chose_indices(self):
        sampled_indices = []
        # Gather K examples from each class
        for i in range(self.num_classes):
            locs = np.argwhere(np.isclose(self.targets, i)).ravel()
            k_sampled = np.random.choice(locs, size=self.k, replace=False)
            sampled_indices.extend(list(k_sampled))
        return np.array(sampled_indices)

    def _get_indices(self):
        # Shuffle, since classes are now in order
        np.random.shuffle(self.indices)
        return self.indices

if __name__ == '__main__':
    from src.data.cifar10 import get_cifar10_dataloaders
    k = 100
    n = 5

    sampler = partial(KUniform, k=k)
    train, test = get_cifar10_dataloaders(sampler=sampler, batch_size=32)
    per_train_y = []
    for _ in range(n):
        per_epoch_y = []
        for i, (x,y) in enumerate(train):
            per_epoch_y.extend(list(y.detach().cpu().numpy()))
        per_train_y.append(per_epoch_y)

    per_train_y = np.array(per_train_y)
    counts = np.array(np.unique(per_train_y, return_counts=True)[1])

    assert len(np.unique(per_train_y, axis=0)) > 1, "There is no shuffling"
    assert np.all(counts), "Data is nott well distributed."

    print("Works fine.")