#!/bin/bash

# Building RDR Rules
python scrdr_interactive.py --input ./data/loan_approval.csv --target Approved \
--exclude ID --tree loan_rules_demo.json --mode build | tee running_inter_loan_demo.log

# Testing with RDR Model
python scrdr_interactive.py --input ./data/loan_approval.csv --target Approved \
--exclude ID --tree loan_rules_demo.json --mode test
