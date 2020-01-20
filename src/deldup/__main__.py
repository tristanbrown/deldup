import argparse
from .experiment import Experiment

parser = argparse.ArgumentParser()
parser.add_argument('path', help='Path to the input data.')
args = parser.parse_args()

def run():
    case = Experiment(args.path)
    case.analyze()

if __name__ == "__main__":
    run()
