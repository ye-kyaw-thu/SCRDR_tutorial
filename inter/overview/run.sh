#!/bin/bash

## the overview of scrdr_interactive.py code
dot -Tpdf ./scrdr_interactive_overview.dot -o ./scrdr_interactive_overview.pdf
dot -Gdpi=300 -Tpng ./scrdr_interactive_overview.dot -o ./scrdr_interactive_overview.png

## the SCRDR rule building with students dataset
dot -Tpdf ./case_with_students_dataset.dot -o ./case_with_students_dataset.pdf
dot -Gdpi=300 -Tpng ./case_with_students_dataset.dot -o ./case_with_students_dataset.png

## student SCRDR model
dot -Tpdf students_model.dot -o students_model.pdf
dot -Gdpi=300 -Tpng students_model.dot -o students_model.png

## the SCRDR rule building with loan dataset
dot -Tpdf ./case_with_loan_dataset.dot -o ./case_with_loan_dataset.pdf
dot -Gdpi=300 -Tpng ./case_with_loan_dataset.dot -o ./case_with_loan_dataset.png

## the SCRDR rule building with loan dataset
dot -Tpdf ./case_with_weather_dataset.dot -o ./case_with_weather_dataset.pdf
dot -Gdpi=300 -Tpng ./case_with_weather_dataset.dot -o ./case_with_weather_dataset.png


