#!/bin/bash

mkdir results_scrdr;

time python ./scrdr_learner.py --input ./data/iris.data --target class \
--plot ./results_scrdr/iris_rdr_cf.png --output  ./results_scrdr/iris_rdr_model.json

time python ./scrdr_learner.py --input ./data/titanic1.csv --target Survived \
--exclude PassengerId --plot  ./results_scrdr/titanic_rdr_cf.png \
--output  ./results_scrdr/titanic_rdr_model.json

time python ./scrdr_learner.py --input ./data/wine.data --target Class \
--plot  ./results_scrdr/wine_rdr_cf.png --output  ./results_scrdr/wine_rdr_model.json

time python ./scrdr_learner.py --input ./data/agaricus-lepiota.data --target class \
--plot  ./results_scrdr/mushroom_rdr_cf.png --output  ./results_scrdr/mushroom_rdr_model.json

time python ./scrdr_learner.py --input ./data/wdbc.data --target Diagnosis \
--exclude "ID" --plot  ./results_scrdr/breast_cancer_rdr_cf.png \
--output  ./results_scrdr/breast_cancer_rdr_model.json

