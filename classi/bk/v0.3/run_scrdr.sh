#!/bin/bash

time python ./scrdr_learner.py --input ./data/iris.data --target class \
--plot iris_rdr_cf.png --output iris_rdr_model.json

time python ./scrdr_learner.py --input ./data/titanic1.csv --target Survived \
--exclude PassengerId --plot titanic_rdr_cf.png --output titanic_rdr_model.json

time python ./scrdr_learner.py --input ./data/wine.data --target Class \
--plot wine_rdr_cf.png --output wine_rdr_model.json

time python ./scrdr_learner.py --input ./data/agaricus-lepiota.data --target class \
--plot mushroom_rdr_cf.png --output mushroom_rdr_model.json

time python ./scrdr_learner.py --input ./data/wdbc.data --target Diagnosis \
--exclude "ID" --plot breast_cancer_rdr_cf.png --output breast_cancer_rdr_model.json

