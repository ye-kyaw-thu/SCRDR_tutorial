#!/bin/bash

# Train with Myanmar syllable-level BIES
python scrdr_tokenizer.py train \
    --train ./data/mypos_v3.word \
    --model model/rdrBurSeg_syl \
    --jobs 20 \
    --threshold-imp 2 \
    --threshold-match 2 \
    --scheme BIES \
    --syllable 

# Evaluate
time python scrdr_tokenizer.py test \
    --input ./data/10k_test.txt \
    --model model/rdrBurSeg_syl \
    --output test_syl.hyp \
    --confusion-matrix syl_bies_cm.png

# Segment raw Burmese text
time python scrdr_tokenizer.py segment \
    --input ./data/10k_test.input \
    --model model/rdrBurSeg_syl \
    --output segmented_syl.txt \
    --separator " "

