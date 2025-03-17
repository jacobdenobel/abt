"""
Extra req:
    sudo apt install bc
    cat ~/.octaverc 
        pkg load signal
        pkg load general
        pkg load control
    
"""

import os
import subprocess
import argparse
import shutil

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def call_fade(*args, capture_output=False, base_folder=None):
    command = ["fade"] + list(args)
    print("running: ", " ".join(command))
    proc = subprocess.run(command, capture_output=capture_output, cwd=base_folder)
    assert proc.returncode == 0
    return proc


def check_fade_installed():
    call_fade("info", capture_output=True)


def check_and_normalize_path(path, create=False):
    if not os.path.exists(path):
        if not create:
            raise FileNotFoundError(path)
        os.makedirs(path)
    return os.path.abspath(path)


def get_arguments(proc_type: str):
    if proc_type in ("speech", "digit",):
        return ["120", "120", "-26:1:6", "[0.5 0.5]"]
    raise NotImplemented


def copy_sources(speech_folder: str, noise_file: str, project_folder: str):
    s_tgt = check_and_normalize_path(os.path.join(project_folder, "source/speech"))
    n_tgt = check_and_normalize_path(os.path.join(project_folder, "source/noise"))

    shutil.copyfile(noise_file, os.path.join(n_tgt, os.path.basename(noise_file)))
    for src in os.listdir(speech_folder):
        source_file = os.path.join(speech_folder, src)
        if os.path.isfile(source_file):
            shutil.copyfile(source_file, os.path.join(s_tgt, src))


def get_summary(project_folder):
    summary_file = os.path.join(project_folder, "evaluation/summary")
    if not os.path.isfile(summary_file):
        print(summary_file, "not found")
        return 
    
    summary = pd.read_csv(summary_file, sep=' ', names=[
        "train_condition", "test_condition",
        "n_words", "n_words_correct",
        "n_sentence", "n_sentence_correct",
        "weighted", 
        "1", "2", "3", "4", "5", "6", "7"
    ])
    
    summary['test_snr'] = summary['test_condition'].str[-3:].astype(int)
    summary['train_snr'] = summary['train_condition'].str[-3:].astype(int)
    summary['fraction_sentence_correct'] = summary['n_sentence_correct']/summary['n_sentence']
    summary['fraction_word_correct'] = summary['n_words_correct']/summary['n_words']
    return summary
    
def show_results(project_folder):
    summary = get_summary(project_folder)
    print(pd.DataFrame(summary.groupby("test_snr")[['fraction_word_correct', "fraction_sentence_correct"]].mean()).T)
    print()
    return summary    
    
def get_folders(args):
    base_folder = check_and_normalize_path(args.base_folder)
    project_folder = os.path.join(base_folder, args.project_name)
    return base_folder, project_folder    

def fade(args):
    base_folder, project_folder = get_folders(args)
    if args.create:
        if os.path.exists(project_folder):
            shutil.rmtree(project_folder)

        print(project_folder)
        
        call_fade(args.project_name, base_folder=base_folder)
        call_fade(
            args.project_name,
            f"parallel",
            *["1"] + [str(args.ncpu) for _ in range(4)],
            base_folder=base_folder,
        )
        call_fade(
            args.project_name,
            f"corpus-{args.proc_type}",
            *get_arguments(args.proc_type),
            base_folder=base_folder,
        )
    if not args.process:
        return 
    # return 
    # copy_sources(
    #     check_and_normalize_path(args.speech_folder),
    #     check_and_normalize_path(args.noise_file),
    #     project_folder,
    # )
    for proc in (
        # "corpus-generate",
        "corpus-format",
        "features",
        "training",
        "recognition",
        "evaluation",
        "figures",
    ):
        call_fade(
            args.project_name,
            proc,
            base_folder=base_folder,
        )
        
    show_results(project_folder)
    
    return project_folder


def create_overview(args, cumulative: bool = False):
    cumulator = np.maximum.accumulate if cumulative else lambda x:x
    summarys = []
    f, ((ax1, ax2), (ax3, ax4), (ax5, ax6)) = plt.subplots(3, 2, figsize=(8, 10))
    
    for folder in os.listdir(args.base_folder):
        if folder not in ("din-raw", "din-bruce", "din-ci-all-vocoded"): continue
        project_folder = os.path.join(args.base_folder, folder)
        if os.path.isdir(project_folder) and (summary := get_summary(project_folder)) is not None:
            summary['project_name'] = folder
            summarys.append(summary)
            
            lines = pd.DataFrame(summary.groupby("test_snr")[['fraction_word_correct', "fraction_sentence_correct"]].mean()).T
            ax1.plot(lines.columns.values, cumulator(lines.iloc[0].values), label=f"{folder} avg")
            ax2.plot(lines.columns.values, cumulator(lines.iloc[1].values), label=f"{folder} avg")
            
            best_train_db = summary.groupby("train_snr")['fraction_word_correct'].mean().idxmax()
            lines_best = summary[summary['train_snr'] == best_train_db].groupby("test_snr")[['fraction_word_correct', "fraction_sentence_correct"]].mean().T
            ax3.plot(lines_best.columns.values, cumulator(lines_best.iloc[0].values), label=f"{folder} at {best_train_db}dB")
            ax4.plot(lines_best.columns.values, cumulator(lines_best.iloc[1].values), label=f"{folder} at {best_train_db}dB")
            
            fixed_db = 0 #if 'raw' in folder else -20

            lines_zero = summary[summary['train_snr'] == fixed_db].groupby("test_snr")[['fraction_word_correct', "fraction_sentence_correct"]].mean().T
            ax5.plot(lines_zero.columns.values, cumulator(lines_zero.iloc[0].values), label=f"{folder} at {fixed_db}dB")
            ax6.plot(lines_zero.columns.values, cumulator(lines_zero.iloc[1].values), label=f"{folder} at {fixed_db}dB")
            
    summary = pd.concat(summarys)
    
    for ax in ax1, ax3, ax5:
        ax.set_ylabel("fraction word correct")
    
    for ax in ax2, ax4, ax6:
        ax.set_ylabel("fraction sentence correct")
    
    for ax in ax1, ax2, ax3, ax4, ax5, ax6:
        ax.set_xlabel("test SNR (dB)")
        ax.grid()
        ax.legend(loc='upper left', fontsize=8)
        
    plt.tight_layout()
    plt.savefig("overview.pdf")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("project_name",  nargs='?', default=None)
    parser.add_argument("speech_folder", nargs='?',  default=None)
    parser.add_argument("noise_file", nargs='?', default=None)
    parser.add_argument("--base_folder", type=str, default="./fade_projects")
    parser.add_argument("--proc_type", type=str, default="speech")
    parser.add_argument("--ncpu", type=int, default=64)
    parser.add_argument("--cleanup", action="store_true")
    parser.add_argument("--show_only_res", action="store_true")
    parser.add_argument("--create", action="store_true")
    parser.add_argument("--process", action="store_true")
    args = parser.parse_args()

    if args.show_only_res:
        if args.project_name is None:
           create_overview(args)            
        else:
            show_results(args)
        exit()
    
    try:
        project_folder = fade(args)
    except Exception as err:
        print(err)

    if args.cleanup:
        shutil.rmtree(project_folder)
