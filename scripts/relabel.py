#!/usr/bin/env python3
import argparse as ap
import xml.etree.ElementTree as ET
from os import listdir
from os.path import isfile, join

translation = {
    "BEFORE"       : "1",       # e < e'
    "AFTER"        : "1",       # e' < e
    "INCLUDES"     : "-1",      # e di e'
    "IS_INCLUDED"  : "-1",      # e d e'
    "DURING"       : "-1",      # e d e'
    "SIMULTANEOUS" : "-1",      # e = e'
    "IAFTER"       : "0",       # e mi e'
    "IBEFORE"      : "0",       # e m e'
    "IDENTITY"     : "-1",      # e = e'
    "BEGINS"       : "-1",      # e s e'
    "ENDS"         : "-1",      # e f e'
    "BEGUN_BY"     : "-1",      # e si e'
    "ENDED_BY"     : "-1",      # e fi e'
    "DURING_INV"   : "-1"       # e di e'
}

TLINK_TAG = "TLINK"

RELATION_ATTRIBUTE_NAME = "relType"
def relabel_file(input_filename, output_filename=None):
    root = ET.parse(input_filename).getroot()
    for child in root:
        if child.tag == TLINK_TAG:
            relation = child.get(RELATION_ATTRIBUTE_NAME)
            child.set(RELATION_ATTRIBUTE_NAME, translation[relation])
    if output_filename == None:
        ouput_filename = input_filename+"2"
    print(root)
    ET.ElementTree(root).write(output_filename, encoding="utf-8")

def main(args):
    if args.recursive:
        folder = args.input_src
        files  = [f for f in listdir(folder) if isfile(join(folder,f))]
        files_w_folder = [join(folder,f) for f in listdir(folder) if isfile(join(folder,f))]
        for i, f in enumerate(files_w_folder):
            outfile = None
            if args.output_file == None:
                outfile = f.split(".")[0]+".new.tml"
            else:
                outfile = join(args.output_file, files[i])
            relabel_file(f, outfile)
    else:
        relabel_file(args.input_src, args.output_file)

if __name__ == "__main__":
    parser = ap.ArgumentParser(description="TimeML ReLabel")
    parser.add_argument('input_src', type=str)
    parser.add_argument('-r', '--recursive', action='store_true', default=False)
    parser.add_argument('-o', '--output-file', type=str, default=None)
    args = parser.parse_args()
    main(args)
