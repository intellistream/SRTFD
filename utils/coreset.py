import numpy as np
from sklearn.metrics import pairwise_distances


class CoresetGreedy:
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

    def sample(self, sample_ratio, buffer):
        sample_size = int(self.dset_size * sample_ratio)
        print("sample_size:{}".format(sample_size))

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
            #print(new_batch)


        _, y = buffer.retrieve()

        y = y.cpu().numpy()

        existing_labels = {label: [] for label in np.unique(y)}

        for label in y:
            existing_labels[label].append(label)

        all_labels = set(new_batch.keys()).union(existing_labels.keys())
        min_count = min(
            len(new_batch.get(label, [])) + len(existing_labels.get(label, []))
            for label in all_labels
        )


        new_balanced_batch = np.array([item for sublist in new_batch.values()
                                    for item in sublist[:min_count]])

        return new_balanced_batch
