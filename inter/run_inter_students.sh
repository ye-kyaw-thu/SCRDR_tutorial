#!/bin/bash

## Building RDR Model
python scrdr_interactive.py --input ./data/students.csv --target Pass_Fail \
--exclude Student_ID --tree students_rules_demo.json --mode build | tee running_inter_students_demo.log

## Testing
python scrdr_interactive.py --input ./data/students.csv --target Pass_Fail \
--exclude Student_ID --tree students_rules_demo.json --mode test
