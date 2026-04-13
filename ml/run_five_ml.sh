#!/bin/bash

mkdir -p results_ml

# Updated exclusion for Titanic to remove non-numeric unique strings (Name, Ticket, Cabin)
datasets=(
    "data/iris.data|class|"
    "data/wine.data|Class|"
    "data/agaricus-lepiota.data|class|"
    "data/wdbc.data|Diagnosis|ID"
    "data/titanic1.csv|Survived|PassengerId Name Ticket Cabin"
)

methods=("dt" "rf" "svm" "nb" "lr")

echo "Starting Exhaustive Benchmark..."

for ds_info in "${datasets[@]}"; do
    IFS='|' read -r file target exclude <<< "$ds_info"
    
    base_name=$(basename "$file" .data)
    base_name=$(basename "$base_name" .csv)

    echo "===================================================="
    echo "Processing Dataset: $file"
    echo "===================================================="

    for m in "${methods[@]}"; do
        output_plot="results_ml/${base_name}_${m}.png"
        
        # We use $exclude without quotes here so bash passes them as multiple arguments
        python ./five_ml.py --input "$file" --target "$target" --method "$m" --exclude $exclude --plot "$output_plot"
    done
done

echo "Exhaustive benchmark complete. Check results_ml/ for 25 total images."

