from argparse import ArgumentParser
from glob import glob
from os.path import join

import pandas as pd
from soundfile import read
from torchmetrics.audio import (
    PerceptualEvaluationSpeechQuality,
    ShortTimeObjectiveIntelligibility,
)
from tqdm import tqdm

from sgmse.util.other import energy_ratios, mean_std

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--test_dir",
        type=str,
        required=True,
        help="Directory containing the original test data (must have subdirectories clean/ and noisy/)",
    )
    parser.add_argument(
        "--enhanced_dir",
        type=str,
        required=True,
        help="Directory containing the enhanced data",
    )
    args = parser.parse_args()

    test_dir = args.test_dir
    clean_dir = join(test_dir, "clean/")
    noisy_dir = join(test_dir, "noisy/")
    enhanced_dir = args.enhanced_dir

    data = {
        "filename": [],
        "pesq": [],
        "estoi": [],
        "si_sdr": [],
        "si_sir": [],
        "si_sar": [],
    }
    sr = 16000

    # Evaluate standard metrics
    noisy_files = sorted(glob("{}/*.wav".format(noisy_dir)))
    for noisy_file in tqdm(noisy_files):
        filename = noisy_file.split("/")[-1]
        x, _ = read(join(clean_dir, filename))
        y, _ = read(noisy_file)
        n = y - x
        x_method, _ = read(join(enhanced_dir, filename))

        wb_pesq = PerceptualEvaluationSpeechQuality(sr, "wb")
        stoi = ShortTimeObjectiveIntelligibility(sr, True)

        data["filename"].append(filename)
        data["pesq"].append(wb_pesq(x, x_method))
        data["estoi"].append(stoi(x, x_method))
        data["si_sdr"].append(energy_ratios(x_method, x, n)[0])
        data["si_sir"].append(energy_ratios(x_method, x, n)[1])
        data["si_sar"].append(energy_ratios(x_method, x, n)[2])

    # Save results as DataFrame
    df = pd.DataFrame(data)

    # Print results
    print(enhanced_dir)
    print("PESQ: {:.2f} ± {:.2f}".format(*mean_std(df["pesq"].to_numpy())))
    print("ESTOI: {:.2f} ± {:.2f}".format(*mean_std(df["estoi"].to_numpy())))
    print("SI-SDR: {:.1f} ± {:.1f}".format(*mean_std(df["si_sdr"].to_numpy())))
    print("SI-SIR: {:.1f} ± {:.1f}".format(*mean_std(df["si_sir"].to_numpy())))
    print("SI-SAR: {:.1f} ± {:.1f}".format(*mean_std(df["si_sar"].to_numpy())))

    # Save DataFrame as csv file
    df.to_csv(join(enhanced_dir, "_results.csv"), index=False)
