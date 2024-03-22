#!/bin/bash

# For example, use this command for AMI train test split:
# bash ../../scripts/split_data.sh /data/shared/datasets/nlp/ami-corpus/topics /data/shared/datasets/nlp/ami/train /data/shared/datasets/nlp/ami/test "ES2004a.json ES2004b.json ES2004c.json ES2004d.json ES2014a.json ES2014b.json ES2014c.json ES2014d.json IS1009a.json IS1009b.json IS1009c.json IS1009d.json TS3003a.json TS3003b.json TS3003c.json TS3003d.json TS3007a.json TS3007b.json TS3007c.json TS3007d.json"

# Source folder containing the files
source_folder="$1"

# Destination folders
train_folder="$2"
test_folder="$3"

# Create destination folders if they don't exist
mkdir -p "$train_folder"
mkdir -p "$test_folder"

# Set the percentage of files to allocate for training (change as needed)
train_percentage=90

# List of filenames to go into the test folder (space-separated)
test_filenames="$4"

# Get the list of files in the source folder
files=("$source_folder"/*)

# Get the total number of files
total_files=${#files[@]}

# Calculate the number of files for training and testing
num_train_files=$((total_files * train_percentage / 100))
num_test_files=$((total_files - num_train_files))

# Create an array of filenames to copy to the test folder
IFS=' ' read -ra test_files <<< "$test_filenames"

# Shuffle the remaining files (non-test files) randomly
shuffled_files=($(comm -23 <(printf '%s\n' "${files[@]}" | sort) <(printf '%s\n' "${test_files[@]}" | sort) | shuf))

# Copy files to the train folder
for ((i = 0; i < num_train_files; i++)); do
    cp "${shuffled_files[$i]}" "$train_folder"
done

# Copy files specified for testing to the test folder
for file in "${test_files[@]}"; do
    cp "$source_folder/$file" "$test_folder"
done

echo "File splitting completed!"
