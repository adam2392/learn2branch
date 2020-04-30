import argparse


def _get_lines(fpath):
    lines = []
    with open(fpath) as fin:
        for line in fin:
            if line.startswith("#SBATCH"):
                lines.append(line)
    return lines

def check_sbatch_file(fpath):
    # find lines with sbatch
    sbatch_lines = _get_lines(fpath)

    problematic_dashes = {}
    for line in sbatch_lines:
        if "–" in line and "-" in line:
            raise RuntimeError(f"Note that you have two types of dashes ('-') "
                               f"in your SBATCH file {fpath} in the same line: \n"
                               f"{line}")
        if "–" in line:
            problematic_dashes["–"] = 1
        if "-" in line:
            problematic_dashes["-"] = 1

        # check other features
    if len(problematic_dashes) == 2:
        raise RuntimeError(f"Note that you have two types of dashes ('-') "
                           f"in your SBATCH file {fpath}.")

    print("Looks like your SBATCH file is okay...")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'fpath',
        help='File path to a SBATCH file (sh file).',
    )
    args = parser.parse_args()

    check_sbatch_file(args.fpath)