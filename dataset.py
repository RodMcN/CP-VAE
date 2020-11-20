import numpy as np
import torch
import h5py
import random


class Dataset(torch.utils.data.IterableDataset):
    def __init__(self, img_files, meas_files=None, train=True, steps=10000, device=None):
        super().__init__()
        self.img_files = img_files
        self.meas_files = meas_files
        self.train = train
        self.steps = steps

        assert len(self.img_files) == len(self.meas_files), f"{len(img_files)} imgs, {len(meas_files)} meas"

    def __iter__(self):
        step = 0
        fileno = 0
        imgno = 0
        data_len = -1

        random_file_indices = torch.randperm(len(self.img_files)).numpy()

        while step < self.steps:

            if imgno >= data_len:
                if fileno >= len(self.img_files)-1:
                    random_file_indices = torch.randperm(len(self.img_files)).numpy()
                    fileno = 0
                else:
                    fileno += 1
                
                try:
                    file_idx = random_file_indices[fileno]
                except:
                    file_idx = random.choice(random_file_indices)
                    

                with h5py.File(self.img_files[file_idx], 'r') as f:
                    for key in f.keys():  # each h5 file is one image, there is only 1 key
                        imgs = f[key][()].astype(np.float32)  # images are 64 bit, convert to 32

                with h5py.File(self.meas_files[file_idx], 'r') as f:
                    for key in f.keys():
                        meas = f[key][()].astype(np.float32)

                data_len = len(imgs)
                imgno = 0

            img = imgs[imgno]
            m = meas[imgno]

            imgno += 1
            if np.any(np.isnan(m)):
                continue
            img = np.nan_to_num(img)

            # augment with random rotations and flipping
            # measurements do not contain any orientation meas so safe to rotate
            # some geometry is in meas so only rotate by 90 degree
            # increments to avoid resizing or cropping
            if self.train:
                img = np.rot90(img, k=np.random.randint(0, 3))
                if torch.rand(1)[0] > 0.5:
                    img = np.fliplr(img)
                if torch.rand(1)[0] > 0.5:
                    img = np.flipud(img)
                # because rotated ndarrays are not supported in pytorch
                img = np.ascontiguousarray(img)

            step += 1

            img = np.moveaxis(img, -1, 0)
            yield torch.Tensor(img), torch.Tensor(m)

def worker_init_fn(worker_id):
    worker_info = torch.utils.data.get_worker_info()
    imgs = worker_info.dataset.img_files
    meas = worker_info.dataset.meas_files

    start = (len(imgs) // worker_info.num_workers) * worker_id
    end = (len(imgs) // worker_info.num_workers) * (worker_id + 1)

    worker_info.dataset.img_files = imgs[start:end]
    worker_info.dataset.meas_files = meas[start:end]