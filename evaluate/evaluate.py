import os
import numpy as np
import collections
from pathlib import Path

def _strip_line(line):
    return line.split("]")[1]

def _read_log_file(fpath):

    summary_txt = []
    train_summary = collections
    with open(fpath, 'r') as fin:
        # read in summary of the training
        line = _strip_line(fin.readline())
        while 'EPOCH 1' not in line:
            summary_txt.append(line)
            line = _strip_line(fin.readline())

        # read train/validation per epoch
        while fin:


    print(summary_txt)
    return summary_txt

if __name__ == '__main__':
    datadir = Path(Path(os.getcwd()).parent / "trained_models")
    modelname = 'baseline'
    problems = ['cauctions', 'indset', 'setcover',  'tsp']
    for problem in problems:
        problem_dir = Path(datadir / problem / modelname )

        seeds = [x for x in problem_dir.glob("*") if not x.name.startswith(".")]
        for seed in seeds:
            seed_dir = Path(problem_dir / seed)
            best_params_fpath = Path(seed_dir / 'best_params.pkl')
            if best_params_fpath in seed_dir.glob("*.pkl"):
                log_fpath = Path(seed_dir / 'log.txt')
                _read_log_file(log_fpath)
                print("done")
        break