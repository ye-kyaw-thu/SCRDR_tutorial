#!/bin/bash

## for training:
time python ./scrdr_tagger.py train --train ./pos_data/corpus.txt --model model/mypos \
--jobs 16 --threshold-imp 2 --threshold-match 2

## for testing:  
time python ./scrdr_tagger.py test --test ./pos_data/otest.txt --model ./model/mypos \
--hyp ./otest.hyp --confusion-matrix pos_cm.png

## for tagging:
time python ./scrdr_tagger.py tag --input ./pos_data/otest.txt --model ./model/mypos \
--output ./otest.tag

## Evaluation
python2.7 ./evaluate.py ./otest.hyp ./pos_data/otest.txt | tee eval.log
