import numpy as np
from sklearn.metrics import pairwise_distances


class Coreset_Greedy:
    def __init__(self, data_pts, p_data_pts, labels, p_labels):

        if p_data_pts.size == 0:
            self.data_pts = data_pts
            self.labels = labels
        else:
            self.data_pts = np.concatenate([data_pts, p_data_pts], axis=0)
            self.labels = np.concatenate([labels, p_labels], axis=0)

        self.data_pts, indices = np.unique(
            self.data_pts, axis=0, return_index=True)
        self.labels = self.labels[indices]

        self.dset_size = len(self.data_pts)

        self.min_distances = np.inf * np.ones(self.dset_size, dtype=np.float32)
        self.already_selected = []

        self.data_pts = self.data_pts.reshape(-1, self.data_pts.shape[1])

    def update_dist(self, centers):
        dist = pairwise_distances(
            self.data_pts[centers], self.data_pts, metric='euclidean')
        self.min_distances = np.minimum(
            self.min_distances, np.min(dist, axis=0))

    def sample(self, sample_ratio, samples_per_class=None):
        sample_size = int(self.dset_size * sample_ratio)

        if samples_per_class:
            sample_size = self.dset_size

        new_batch = {label: [] for label in np.unique(self.labels)}

        for _ in range(sample_size):
            if not self.already_selected:
                ind = np.random.choice(self.dset_size)
            else:
                ind = np.argmax(self.min_distances)

                while ind in self.already_selected:
                    self.min_distances[ind] = 0
                    ind = np.argmax(self.min_distances)

            self.already_selected.append(ind)
            self.update_dist([ind])

            new_batch[self.labels[ind]].append(ind)

        min_count = min(len(lst)for lst in new_batch.values())
        if samples_per_class:
            min_count = min(min_count, samples_per_class)

        new_balanced_batch = np.array([item for sublist in new_batch.values()
                                       for item in sublist[:min_count]])

        return new_balanced_batch
