# Speech Enhancement and Dereverberation with Diffusion-based Generative Models

<img src="https://raw.githubusercontent.com/sp-uhh/sgmse/main/diffusion_process.png" width="500" alt="Diffusion process on a spectrogram: In the forward process noise is gradually added to the clean speech spectrogram x0, while the reverse process learns to generate clean speech in an iterative fashion starting from the corrupted signal xT.">

This repository contains the NONOFFICIAL PyTorch reimplementation for the 2023 papers listed below for my graduation work:

- [*Speech Enhancement and Dereverberation with Diffusion-Based Generative Models*](https://arxiv.org/abs/2208.05830), 2022 [1]
- [*Analysing Diffusion-based Generative Approaches versus Discriminative Approaches for Speech Restoration*](https://arxiv.org/abs/2211.02397), 2023 [2]

I trained model on open source VoiceBank+Demand dataset, made some experiments with generative/discriminative methods and writed about it in my graduation work, pdf version of which could be founded in folder `pdf_thesis`. As for the code, I only resolved some conflicts in new versions of libraries for 3.10 version of Python, fixed codestyle with formatter for readability and made small changes, bringing it all together.

## Installation

- Create a new virtual environment with Python 3.10.
- Install the package dependencies via `pip install -r requirements.txt`.
  - Let pip resolve the dependencies for you. If you encounter any issues, please check `requirements_version.txt` for the exact versions we used.
- If using W&B logging (default):
    - Set up a [wandb.ai](https://wandb.ai/) account
    - Log in via `wandb login` before running our code.
- If not using W&B logging:
    - Pass the option `--nolog` to `train.py`.
    - Your logs will be stored as local CSVLogger logs in `lightning_logs/`.

## Training

Training is done by executing `train.py`. A minimal running example with default settings can be run with

```bash
python train.py --base_dir <your_base_dir>
```

where `your_base_dir` should be a path to a folder containing subdirectories `train/` and `valid/` (optionally `test/` as well). Each subdirectory must itself have two subdirectories `clean/` and `noisy/`, with the same filenames present in both. We currently only support training with `.wav` files.

To see all available training options, run `python train.py --help`. Note that the available options for the SDE and the backbone network change depending on which SDE and backbone you use. These can be set through the `--sde` and `--backbone` options.


## Usage:
- For resuming training, you can use the `--ckpt` option of `train.py`.
- For evaluating these checkpoints, use the `--ckpt` option of `enhancement.py` (see section **Evaluation** below).

## Evaluation

To evaluate on a test set, run
```bash
python enhancement.py --test_dir <your_test_dir> --enhanced_dir <your_enhanced_dir> --ckpt <path_to_model_checkpoint>
```

to generate the enhanced .wav files, and subsequently run

```bash
python calc_metrics.py --test_dir <your_test_dir> --enhanced_dir <your_enhanced_dir>
```

to calculate and output the instrumental metrics.

Both scripts should receive the same `--test_dir` and `--enhanced_dir` parameters. The `--cpkt` parameter of `enhancement.py` should be the path to a trained model checkpoint, as stored by the logger in `logs/`.

>[1] Julius Richter, Simon Welker, Jean-Marie Lemercier, Bunlong Lay, Timo Gerkmann. "Speech Enhancement and Dereverberation with Diffusion-Based Generative Models", IEEE/ACM Transactions on Audio, Speech, and Language Processing, vol. 31, pp. 2351-2364, 2023.
>

>[2] Jean-Marie Lemercier, Julius Richter, Simon Welker and Timo Gerkmann. "Analysing Diffusion-based Generative Approaches versus Discriminative Approaches for Speech Restoration", *ICASSP*, 2023.
>