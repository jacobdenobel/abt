
import os
import argparse
import glob
import subprocess

from run_reconstruction import NEUROGRAM_GENERATORS

BASE_FOLDER = os.path.abspath(os.path.dirname(__file__))

OUTPUT_FOLDER = '/scratch/jacob/output_folder'

def reconstruct(path, output_path, neurogram_generator):
    for _ in range(5):
        cmd = f"python run_reconstruction.py {path} "\
              f"--output_path={output_path} "\
              f"--neurogram_generator={neurogram_generator}"
        res = subprocess.run(
            cmd,
            cwd=BASE_FOLDER,
            shell=True
        )
        if res.returncode == 0:
            break


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=str)
    parser.add_argument("--output_path", type=str, default=None)
    parser.add_argument(
        "--neurogram_generator",
        default=0,
        choices=range(len(NEUROGRAM_GENERATORS)),
        type=int,
    )
    args = parser.parse_args()
    path = os.path.normpath(args.path)
    
    if os.path.isdir(path) and path.endswith("corpus"):
        files = glob.glob(f"{args.path}/*/*/*/*/*.wav")[::-1]
        for i, file in enumerate(files, 1):
            print(f"{i}/{len(files)}: {file.replace(path, '')}")
            output_folder = os.path.join(OUTPUT_FOLDER, os.path.dirname(file))
            reconstruct(file, output_folder, args.neurogram_generator)