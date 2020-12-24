import argparse as ap
from .model import validator as vd
from group20 import plots as pt
from .model import model as md
from .timeml import timemldoc as td
from sklearn.metrics import recall_score
from group20 import experiments as ex
import os
import nltk

def generate_file_paths(d, n=None):
    """ Given directory d, takes the first n files from it """
    paths = []
    for entry in os.scandir(d): 
        if n is None or len(paths) < n:
            paths.append(entry.path)
    return paths


def main(args):
    states = [md.START_st, md.EVENT, md.NEVENT, md.END_st]
    paths = generate_file_paths(args.directory, args.number)

    #print("RAW EMISSION")
    #ex.evaluate_hmm(paths, states, md.raw_e_n, "Raw emissions")
    # ex.confusion_and_pr(paths, states, md.raw_e_n)
    #print("POS TAGS")
    # ex.evaluate_hmm(paths, states, md.pos_e_n, "POS tag conversion")
    # ex.confusion_and_pr(paths, states, md.pos_e_n)

    #ex.plot_smooth()

    #ex.baseline(paths, states, md.raw_e_n)
    ex.plot_baseline_compare()

    
if __name__ == "__main__":
    parser = ap.ArgumentParser(description="ML Project")
    parser.add_argument('-d', '--directory', type=str)
    parser.add_argument('-n', '--number', type=int, default=None)
    args = parser.parse_args()
    main(args)

