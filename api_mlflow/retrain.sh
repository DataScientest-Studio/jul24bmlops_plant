#!/bin/bash

# Check if at least 2 paths are provided
if [ $# -lt 3 ]; then
  echo "Usage: $0 <model_file_path> <path1> <path2> [<path3> ... <pathN>]"
  exit 1
fi

# First argument is the model_file_path
model_file_path=$1
shift  # Remove the model_file_path from the argument list

# Build the query string for paths
query_string=""
for path in "$@"; do
  if [ -n "$query_string" ]; then
    query_string="${query_string}&paths=${path}"
  else
    query_string="paths=${path}"
  fi
done

# Append the model_file_path to the query string
query_string="${query_string}&model_file_path=${model_file_path}"

# Execute the curl command with the constructed query string
curl -X 'POST' \
  "http://localhost:85/retrain?${query_string}" \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d ''
