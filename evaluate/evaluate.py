import os
import numpy as np
import collections
from pathlib import Path

def _strip_line(line):
    try:
        line = line.split("]")[1]
    except:
        line = None
    return line

def _summarize_line(line):
    loss = line.split("LOSS: ")[1].split(" ")[0]
    acc1 = line.split("acc@1: ")[1].split(" ")[0]
    acc3 = line.split("acc@3: ")[1].split(" ")[0]
    acc5 = line.split("acc@5: ")[1].split(" ")[0]
    acc10 = line.split("acc@10: ")[1].split(" ")[0]
    return (loss, acc1, acc3, acc5, acc10)

def _read_log_file(fpath):

    summary_txt = []
    summary_txt.append(['loss', 'acc@1', 'acc@3', 'acc@5', 'acc@10'])
    train_summary = collections.defaultdict(list)
    test_summary = collections.defaultdict(list)

    with open(fpath, 'r') as fin:
        # read in summary of the training
        line = _strip_line(fin.readline())
        while 'EPOCH 1' not in line:
            summary_txt.append(line)
            line = _strip_line(fin.readline())

        epoch_ind = 0
        # read train/validation per epoch
        while line:
            if 'EPOCH' in line:
                epoch_ind += 1
            elif 'TRAIN' in line:
                summary = _summarize_line(line)
                train_summary[epoch_ind].append(summary)
            elif 'VALID' in line:
                summary = _summarize_line(line)
                test_summary[epoch_ind].append(summary)
            line = _strip_line(fin.readline())
    return summary_txt, train_summary, test_summary

if __name__ == '__main__':
    datadir = Path(Path(os.getcwd()).parent / "trained_models")
    modelname = 'baseline'
    problems = ['cauctions', 'indset', 'setcover',  'tsp']
    for problem in problems:
        problem_dir = Path(datadir / problem / modelname )

        seed_dirs = [x for x in problem_dir.glob("*") if not x.name.startswith(".")]
        for seed_dir in seed_dirs:
            print(problem, seed_dir)
            # seed_dir = Path(problem_dir / seed)
            best_params_fpath = Path(seed_dir / 'best_params.pkl')
            if best_params_fpath in seed_dir.glob("*.pkl"):
                log_fpath = Path(seed_dir / 'log.txt')
                summary_txt, train_summary, test_summary = _read_log_file(log_fpath)

                print("done")
                print(train_summary)
                print(test_summary)
        # break