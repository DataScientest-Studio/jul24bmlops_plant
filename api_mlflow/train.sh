#!/bin/bash

# Check if the user provided a parameter
if [ -z "$1" ]; then
  echo "Usage: $0 <paths>"
  exit 1
fi

# Assign the first argument to the variable 'paths'
paths=$1

# Execute the curl command with the 'paths' parameter
curl -X 'POST' \
  "http://localhost:85/train?paths=${paths}" \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d ''
