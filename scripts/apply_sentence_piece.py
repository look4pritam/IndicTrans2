import sentencepiece as spm
import os
import sys
import multiprocessing

def tokenize(lang, spm_model_path, data_dir, bpe_dir, split):
    spm_processor = spm.SentencePieceProcessor(model_file=spm_model_path)
    
    print(f"working with {lang}")
    with open(os.path.join(data_dir, f"{split}.{lang}"), 'r') as infile, \
         open(os.path.join(bpe_dir, f"{split}.{lang}"), 'w') as outfile:
        for line in infile:
            pieces = spm_processor.encode_as_pieces(line.strip())
            outfile.write(' '.join(pieces) + '\n')

if __name__ == '__main__':
    exp_dir = sys.argv[1]
    data_dir = sys.argv[2]
    bpe_dir = sys.argv[3]
    src_lang = sys.argv[4]
    tgt_lang = sys.argv[5]
    split = sys.argv[6]

    # create separate processes for each language
    src_process = multiprocessing.Process(target=tokenize, args=(src_lang, os.path.join(exp_dir, 'vocab', f'model.{src_lang}'), data_dir, bpe_dir, split))
    tgt_process = multiprocessing.Process(target=tokenize, args=(tgt_lang, os.path.join(exp_dir, 'vocab', f'model.{tgt_lang}'), data_dir, bpe_dir, split))

    # start both processes
    src_process.start()
    tgt_process.start()

    # wait for both processes to finish
    src_process.join()
    tgt_process.join()