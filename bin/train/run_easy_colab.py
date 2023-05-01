# File: run_easy_colab.py
# File Created: Saturday, 1st May 2023 8:10:01 pm
# Author: Fábio Silva (silva.fabio@gmail.com)

from argparse import ArgumentParser
from nam.train.colab import run as run_colab

def to_dict(arg_list):
    try:
        return {name: int(str_v) if str_v.isdigit() else float(str_v) if '.' in str_v else str_v for name, str_v in (s.split('=') for s in arg_list)}
    except Exception:
        print(f"Error in argument list: '{arg_list}'")
        print(" ** Example of use: -l epochs=10 architecture=standard delay=10 lr=0.004 lr_decay=0.007 seed=0")
        return None

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--epochs", "-e", type=int, default=100, help="Set epoch when running easy colab training")
    parser.add_argument("--arch", "-a", type=str, default="standard", help="Set architecture when running easy colab training")
    parser.add_argument("--arg-list", "-l", type=str, default=None, nargs='*', help="Set list of parameter values, like so: -l epochs=10 architecture=standard lr=0.004 lr_decay=0.007 seed=0")
    args = parser.parse_args()
    if args.arg_list:
        arg_list=to_dict(args.arg_list)
        if arg_list:
            print(f"* Running easy colab: run({arg_list})")
            run_colab(**arg_list)
    else:
        print(f"* Running easy colab: run(epochs={args.epochs}, architecture={args.arch})")
        run_colab(epochs=args.epochs, architecture=args.arch)
