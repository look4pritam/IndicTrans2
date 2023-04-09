import os
import sys
import sentencepiece as spm
from joblib import Parallel, delayed
from tqdm import tqdm
import multiprocessing

def tokenize(spm_model_path, infname, outfname):
    spm_processor = spm.SentencePieceProcessor(model_file=spm_model_path)
    
    with open(infname, "r") as infile, open(outfname, "w") as outfile:
        for line in tqdm(infile):
            pieces = spm_processor.encode_as_pieces(line.replace("\n", "").strip())
            outfile.write(' '.join(pieces) + '\n')

if __name__ == '__main__':
    spm_model_path = sys.argv[1]
    infname = sys.argv[2]
    outfname = sys.argv[3]
    tokenize(spm_model_path, infname, outfname)