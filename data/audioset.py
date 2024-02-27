import os
import random
import torch
import numpy as np
import data.misc as misc

NUM_CLASSES = 527
BLACK_LIST = [327, 500, 513, 514, 515, 520, 521]


class AudioSetViews(torch.utils.data.dataset.Dataset):
    def __init__(self, df_path, transforms, sample_rate, num_views=2, num_repeat=1, max_size=None):
        self.transforms = transforms
        self.sample_rate = sample_rate
        self.num_views = num_views
        self.num_repeat = num_repeat
        self.root_path = os.path.dirname(df_path)

        self.df = misc.load_df(df_path)

        if max_size is not None:
            self.df = self.df.sample(max_size).reset_index(drop=True)

        self.dfs = dict()
        self.class_indexes = list()
        for i in range(NUM_CLASSES):
            self.dfs[i] = self.df[self.df[i] == 1].reset_index(drop=True)
            if (len(self.dfs[i])) and (not i in BLACK_LIST):
                self.class_indexes.append(i)
        self.shuffle_class_indexes()

    def __getitem__(self, index):
        class_index = self.class_indexes[(index // self.num_repeat) % len(self.class_indexes)]

        std = 0
        while std == 0:
            row = self.dfs[class_index].sample(1).iloc[0]
            path = os.path.join(self.root_path, row["relative"])
            audio = misc.load_audio(path, sample_rate=self.sample_rate)
            std = audio.std()

        labels_index = row["labels_index"]
        labels = self._onehot(labels_index)

        views = list()
        for _ in range(self.num_views):
            views.append(np.expand_dims(self.transforms(audio), axis=0))
        views = np.concatenate(views, axis=0)

        data_dict = {
            "audio": views,
            "target": labels,
        }

        return data_dict

    def _onehot(self, labels):
        one_hot = np.zeros(NUM_CLASSES)
        for label in labels:
            one_hot[label] = 1.0
        return one_hot

    def shuffle_class_indexes(self):
        random.shuffle(self.class_indexes)

    def __len__(self):
        return len(self.df)
