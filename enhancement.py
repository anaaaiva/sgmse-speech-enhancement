from argparse import ArgumentParser
from glob import glob
from os.path import join

import torchaudio.transforms as T
from soundfile import write
from torchaudio import load
from tqdm import tqdm

from sgmse.model import DiscriminativeModel, ScoreModel
from sgmse.util.other import ensure_dir

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--test_dir",
        type=str,
        required=True,
        help="Directory containing the test data (must have subdirectory noisy/)",
    )
    parser.add_argument(
        "--enhanced_dir",
        type=str,
        required=True,
        help="Directory containing the enhanced data",
    )
    parser.add_argument("--ckpt", type=str, help="Path to model checkpoint.")
    parser.add_argument(
        "--corrector",
        type=str,
        choices=("ald", "langevin", "none"),
        default="ald",
        help="Corrector class for the PC sampler.",
    )
    parser.add_argument(
        "--corrector_steps", type=int, default=1, help="Number of corrector steps"
    )
    parser.add_argument(
        "--snr",
        type=float,
        default=0.5,
        help="SNR value for (annealed) Langevin dynmaics.",
    )
    parser.add_argument("--N", type=int, default=30, help="Number of reverse steps")
    parser.add_argument(
        "--discriminatively",
        action="store_true",
        help="Use a discriminative model instead",
    )
    args = parser.parse_args()

    noisy_dir = join(args.test_dir, "noisy/")
    checkpoint_file = args.ckpt
    corrector_cls = args.corrector

    target_dir = args.enhanced_dir
    ensure_dir(target_dir)

    # Settings
    sr = 16000
    snr = args.snr
    N = args.N
    corrector_steps = args.corrector_steps

    model_cls = ScoreModel if not args.discriminatively else DiscriminativeModel
    model = model_cls.load_from_checkpoint(
        args.ckpt, base_dir="", batch_size=8, kwargs=dict(gpu=False), num_workers=0
    )
    model.eval(no_ema=False)
    model.cuda()

    noisy_files = sorted(glob("{}/*.wav".format(noisy_dir)))

    for noisy_file in tqdm(noisy_files):
        filename = noisy_file.split("/")[-1]

        # Load wav
        y, sr = load(noisy_file)
        if sr != 16000:
            resampler = T.Resample(orig_freq=sr, new_freq=16000)
            y = resampler(y)
            sr = 16000
        assert sr == 16000, "Pretrained models worked wth sampling rate of 16000"
        x_hat = model.enhance(
            y,
            corrector=args.corrector,
            N=args.N,
            corrector_steps=args.corrector_steps,
            snr=args.snr,
        )

        # Write enhanced wav file
        write(join(target_dir, filename), x_hat, sr)
