import numpy as np
import h5py
import csv
import time
import logging


class AudioSetDataset(object):
    def __init__(self, transforms=None):
        """This class takes the meta of an audio clip as input, and return
        the waveform and target of the audio clip. This class is used by DataLoader.
        """
        self.transforms = transforms

    def __getitem__(self, meta):
        """Load waveform and target of an audio clip.

        Args:
          meta: {
            'hdf5_path': str,
            'index_in_hdf5': int}

        Returns:
          data_dict: {
            'audio_name': str,
            'waveform': (clip_samples,),
            'target': (classes_num,)}
        """
        hdf5_path = meta["hdf5_path"]
        index_in_hdf5 = meta["index_in_hdf5"]

        with h5py.File(hdf5_path, "r") as hf:
            audio_name = hf["audio_name"][index_in_hdf5].decode()
            target = hf["target"][index_in_hdf5].astype(np.float32)

            waveform = int16_to_float32(hf["waveform"][index_in_hdf5])
            if self.transforms is not None:
                waveform = self.transforms(waveform)

        data_dict = {"audio_name": audio_name, "waveform": waveform, "target": target}

        return data_dict


class AudioSetViewsDataset(object):
    def __init__(self, transforms, num_views=2):
        """This class takes the meta of an audio clip as input, and return
        the waveform and target of the audio clip. This class is used by DataLoader.
        """
        self.transforms = transforms
        self.num_views = num_views

    def __getitem__(self, meta):
        """Load waveform and target of an audio clip.

        Args:
          meta: {
            'hdf5_path': str,
            'index_in_hdf5': int}

        Returns:
          data_dict: {
            'audio_name': str,
            'waveform': (clip_samples,),
            'target': (classes_num,)}
        """
        hdf5_path = meta["hdf5_path"]
        index_in_hdf5 = meta["index_in_hdf5"]

        with h5py.File(hdf5_path, "r") as hf:
            audio_name = hf["audio_name"][index_in_hdf5].decode()
            target = hf["target"][index_in_hdf5].astype(np.float32)

            waveform = int16_to_float32(hf["waveform"][index_in_hdf5])

            views = list()
            for _ in range(self.num_views):
                views.append(self.transforms(waveform))

        data_dict = {"audio_name": audio_name, "waveform": views, "target": target}

        return data_dict


class Base(object):
    def __init__(self, indexes_hdf5_path, batch_size, black_list_csv, random_seed):
        """Base class of train sampler.

        Args:
          indexes_hdf5_path: string
          batch_size: int
          black_list_csv: string
          random_seed: int
        """
        self.batch_size = batch_size
        self.random_state = np.random.RandomState(random_seed)

        # Black list
        if black_list_csv:
            self.black_list_names = read_black_list(black_list_csv)
        else:
            self.black_list_names = []

        logging.info("Black list samples: {}".format(len(self.black_list_names)))

        # Load target
        load_time = time.time()

        with h5py.File(indexes_hdf5_path, "r") as hf:
            self.audio_names = [audio_name.decode() for audio_name in hf["audio_name"][:]]
            self.hdf5_paths = [hdf5_path.decode() for hdf5_path in hf["hdf5_path"][:]]
            self.indexes_in_hdf5 = hf["index_in_hdf5"][:]
            self.targets = hf["target"][:].astype(np.float32)

        (self.audios_num, self.classes_num) = self.targets.shape
        logging.info("Training number: {}".format(self.audios_num))
        logging.info("Load target time: {:.3f} s".format(time.time() - load_time))


class MultipleBalancedSampler(Base):
    def __init__(
        self,
        indexes_hdf5_path,
        batch_size,
        black_list_csv=None,
        num_repeat=2,
        random_seed=1234,
    ):
        """Balanced sampler. Generate batch meta for training. Data are equally
        sampled from different sound classes.

        Args:
          indexes_hdf5_path: string
          batch_size: int
          black_list_csv: string
          random_seed: int
        """
        super(MultipleBalancedSampler, self).__init__(
            indexes_hdf5_path, batch_size, black_list_csv, random_seed
        )
        self.num_repeat = num_repeat
        self.samples_num_per_class = np.sum(self.targets, axis=0)
        logging.info(
            "samples_num_per_class: {}".format(self.samples_num_per_class.astype(np.int32))
        )

        # Training indexes of all sound classes. E.g.:
        # [[0, 11, 12, ...], [3, 4, 15, 16, ...], [7, 8, ...], ...]
        self.indexes_per_class = []

        for k in range(self.classes_num):
            self.indexes_per_class.append(np.where(self.targets[:, k] == 1)[0])

        # Shuffle indexes
        for k in range(self.classes_num):
            self.random_state.shuffle(self.indexes_per_class[k])

        self.class_black_list = [327, 500, 513, 514, 515, 520, 521]
        self.queue = []
        self.pointers_of_classes = [0] * self.classes_num

    def expand_queue(self, queue):
        classes_set = np.arange(self.classes_num).tolist()
        self.random_state.shuffle(classes_set)
        queue += classes_set
        return queue

    def __iter__(self):
        """Generate batch meta for training.

        Returns:
          batch_meta: e.g.: [
            {'hdf5_path': string, 'index_in_hdf5': int},
            ...]
        """
        batch_size = self.batch_size

        while True:
            batch_meta = []
            i = 0
            while i < batch_size:
                if len(self.queue) == 0:
                    self.queue = self.expand_queue(self.queue)

                class_id = self.queue.pop(0)
                if class_id in self.class_black_list:
                    continue
                j = 0
                while i < batch_size and j < self.num_repeat:
                    pointer = self.pointers_of_classes[class_id]
                    self.pointers_of_classes[class_id] += 1
                    index = self.indexes_per_class[class_id][pointer]

                    # When finish one epoch of a sound class, then shuffle its indexes and reset pointer
                    if self.pointers_of_classes[class_id] >= self.samples_num_per_class[class_id]:
                        self.pointers_of_classes[class_id] = 0
                        self.random_state.shuffle(self.indexes_per_class[class_id])

                    # If audio in black list then continue
                    if self.audio_names[index] in self.black_list_names:
                        continue
                    else:
                        batch_meta.append(
                            {
                                "hdf5_path": self.hdf5_paths[index],
                                "index_in_hdf5": self.indexes_in_hdf5[index],
                            }
                        )
                        i += 1
                        j += 1

            yield batch_meta

    def state_dict(self):
        state = {
            "indexes_per_class": self.indexes_per_class,
            "queue": self.queue,
            "pointers_of_classes": self.pointers_of_classes,
        }
        return state

    def load_state_dict(self, state):
        self.indexes_per_class = state["indexes_per_class"]
        self.queue = state["queue"]
        self.pointers_of_classes = state["pointers_of_classes"]


def collate_fn(list_data_dict):
    """Collate data.
    Args:
      list_data_dict, e.g., [{'audio_name': str, 'waveform': (clip_samples,), ...},
                             {'audio_name': str, 'waveform': (clip_samples,), ...},
                             ...]
    Returns:
      np_data_dict, dict, e.g.,
          {'audio_name': (batch_size,), 'waveform': (batch_size, clip_samples), ...}
    """
    np_data_dict = {}

    for key in list_data_dict[0].keys():
        np_data_dict[key] = np.array([data_dict[key] for data_dict in list_data_dict])

    return np_data_dict


def read_black_list(black_list_csv):
    """Read audio names from black list."""
    with open(black_list_csv, "r") as fr:
        reader = csv.reader(fr)
        lines = list(reader)

    black_list_names = ["Y{}.wav".format(line[0]) for line in lines]
    return black_list_names


def float32_to_int16(x):
    assert np.max(np.abs(x)) <= 1e6
    x = np.clip(x, -1, 1)
    return (x * 32767.0).astype(np.int16)


def int16_to_float32(x):
    return (x / 32767.0).astype(np.float32)
