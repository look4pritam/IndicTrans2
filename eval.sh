#!/bin/bash

echo `date`
devtest_data_dir=$1
ckpt_dir=$2
model=$3

pairs=$(ls -d $devtest_data_dir/*)

for pair in ${pairs[@]}; do
    pair=$(basename $pair)
    src_lang=$(echo "$pair" | cut -d "-" -f 1)
    tgt_lang=$(echo "$pair" | cut -d "-" -f 2)

    src_fname=$devtest_data_dir/$src_lang-$tgt_lang/test.$src_lang
    tgt_fname=$devtest_data_dir/$src_lang-$tgt_lang/test.$tgt_lang

    if [ -f "$src_fname" ] && [ -f "$tgt_fname" ]; then
        echo "Evaluating $src_lang-$tgt_lang ..."
    else 
        echo "Skipping $src_lang-$tgt_lang ..."
        continue
    fi

    echo "Generating Translations"
    bash joint_translate.sh $src_fname $tgt_fname.pred.itv2 $src_lang $tgt_lang $ckpt_dir $model

    echo "Computing Metrics"
    bash compute_bleu.sh $tgt_fname.pred.itv2 $tgt_fname $tgt_lang > $devtest_data_dir/$src_lang-$tgt_lang/${src_lang}-${tgt_lang}_itv2_scores.json

    rm $tgt_fname.* $src_fname.*
done

echo "Converting results to csv format"
python json_to_csv.py $devtest_data_dir $model