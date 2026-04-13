#!/bin/bash

dot -Tpdf scrdr_learner.dot -o scrdr_learner.pdf
dot -Gdpi=300 -Tpng scrdr_learner.dot -o scrdr_learner.png
