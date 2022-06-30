import numpy as np
import glob
from soundfile import read, write
from tqdm import tqdm
from pesq import pesq
from torchaudio import load
import torch
from argparse import ArgumentParser

from sgmse.data_module import SpecsDataModule
from sgmse.sdes import OUVESDE
from sgmse.model import ScoreModel

from utils import pad_spec, ensure_dir


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--test", type=str, help="Specify test set.")
    parser.add_argument("--train", type=str, help="Specify train set.")
    parser.add_argument("--test_dir", type=str, required=True, help='Directory containing the test data')
    parser.add_argument("--ckpt", type=str,  help='Path to model checkpoint.')
    parser.add_argument("--corrector", type=str, choices=("ald", "none"), default="ald", 
        help="Corrector class for the PC sampler.")
    parser.add_argument("--corrector_steps", type=int, default=1, help="Number of corrector steps")
    parser.add_argument("--snr", type=float, default=0.33, help="SNR value for annealed Langevin dynmaics.")
    parser.add_argument("--N", type=int, default=30, help="Number of reverse steps")
    args = parser.parse_args()

    noisy_dir = args.test_dir
    checkpoint_file = args.ckpt
    corrector_cls = args.corrector

    target_dir = "/export/home/jrichter/repos/sgmse/enhanced/test_{}/train_{}/".format(
        args.test, args.train) 

    ensure_dir(target_dir)

    # Settings
    sr = 16000
    snr = args.snr
    N = args.N
    corrector_steps = args.corrector_steps

    # Load score model 
    model = ScoreModel.load_from_checkpoint(
        checkpoint_file, base_dir='/export/home/jrichter/data/wsj0_chime3/',
        batch_size=16, num_workers=0, kwargs=dict(gpu=False)
    )
    model.eval(no_ema=False)
    model.cuda()

    noisy_files = sorted(glob.glob('{}/*.wav'.format(noisy_dir)))

    for noisy_file in tqdm(noisy_files):
        filename = noisy_file.split('/')[-1]
        
        # Load wav
        y, _ = load(noisy_file) 
        T_orig = y.size(1)   

        # Normalize
        norm_factor = y.abs().max()
        y = y / norm_factor
        
        # Prepare DNN input
        Y = torch.unsqueeze(model._forward_transform(model._stft(y.cuda())), 0)
        Y = pad_spec(Y)
        
        # Reverse sampling
        sampler = model.get_pc_sampler(
            'reverse_diffusion', corrector_cls, Y.cuda(), N=N, 
            corrector_steps=corrector_steps, snr=snr)
        sample = sampler()
        
        # Backward transform in time domain
        x_hat = model.to_audio(sample.squeeze(), T_orig)

        # Renormalize
        x_hat = x_hat * norm_factor

        # Write enhanced wav file
        write(target_dir+filename, x_hat.cpu().numpy(), 16000)
        
        