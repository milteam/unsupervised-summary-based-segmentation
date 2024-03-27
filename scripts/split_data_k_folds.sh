#!/bin/bash

source_dir="$1"
dest_dir="$2"

mkdir -p "$dest_dir"

# Set the number of folds (k) for cross-validation
k=5

# Get the list of files in the source folder
files=("$source_dir"/*)

# Calculate the number of files per fold
files_per_fold=$(( ${#files[@]} / k ))

# Shuffle the files randomly
shuf_files=($(shuf -e "${files[@]}"))  # if you want to add randomness to the splitting
# shuf_files=("${files[@]}")

# Create k-fold cross-validation folders
for ((fold=1; fold<=k; fold++))
do
    # Create the fold directory
    fold_dir="$dest_dir/fold$fold"
    mkdir -p "$fold_dir"

    # Calculate the start and end indices for the files in the current fold
    start=$(( (fold - 1) * files_per_fold ))
    end=$(( start + files_per_fold ))

    # Create the train and test subfolders for the current fold
    train_dir="$fold_dir/train"
    test_dir="$fold_dir/test"
    mkdir -p "$train_dir"
    mkdir -p "$test_dir"

    # Move files to the train subfolder
    for ((i=0; i<start; i++))
    do
        cp "${shuf_files[i]}" "$train_dir"
    done

    # Move files to the test subfolder
    for ((i=start; i<end; i++))
    do
        cp "${shuf_files[i]}" "$test_dir"
    done

    # Move remaining files to the train subfolder
    for ((i=end; i<${#shuf_files[@]}; i++))
    do
        cp "${shuf_files[i]}" "$train_dir"
    done
done

echo "Cross-validation folders created successfully."
