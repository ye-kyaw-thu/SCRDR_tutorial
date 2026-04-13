#!/bin/bash

# Building RDR Rules
python scrdr_interactive.py --input ./data/weather_activity.csv --target Activity \
--exclude Day --tree weather_demo.json --mode build | tee running_inter_weather_demo.log

# Testing with RDR Model
python scrdr_interactive.py --input ./data/weather_activity.csv --target Activity \
--exclude Day --tree weather_demo.json --mode test
