from CPVAE import *
from dataset import Dataset
from train import train
import glob

img_files = sorted(glob.glob('/nfs/cpv/split_and_merged/*.hdf5'))[:-100]
meas_files = sorted(glob.glob('/nfs/cpv/split_and_merged/meas/*.hdf5'))[:-100]

train_dataset = Dataset(img_files, meas_files)
train_data_loader = torch.utils.data.DataLoader(train_dataset, batch_size=16, num_workers=4)

model = VAE(Encoder(latent_dims=144), Decoder(latent_dims=144, num_featues=672))
opt = torch.optim.adam(model.parameters(), lr=0.00005)
train(model, opt, 25, train_data_loader, annealing=6, kl_cap=2, scale_kl=100, save_path='/nfs/cpv/models')

import subprocess
subprocess.run(['chmod', '-R', '770', '~/*'])