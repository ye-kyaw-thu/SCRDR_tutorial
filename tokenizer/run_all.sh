#!/bin/bash

# Train (your real Myanmar myPOS corpus)
time python scrdr_tokenizer.py train \
    --train ./data/mypos_v3.word \
    --model model/rdrBurSeg \
    --jobs 20 \
    --threshold-imp 2 \
    --threshold-match 2 \
    --scheme BIES

# Evaluate
time python scrdr_tokenizer.py test \
    --input ./data/10k_test.txt \
    --model model/rdrBurSeg \
    --output test.hyp \
    --confusion-matrix bies_cm.png

# Segment raw Burmese text
time python scrdr_tokenizer.py segment \
    --input ./data/10k_test.input \
    --model model/rdrBurSeg \
    --output segmented.txt \
    --separator " "

