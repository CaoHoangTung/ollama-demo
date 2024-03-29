#!/bin/bash

# Define the path to the PDF files
pdf_path="./pdf"
model="llama2"

# Iterate over each file in the PDF path
for file in "$pdf_path"/*; do
    # Check if the file is a regular file
    if [ -f "$file" ]; then
        # Run the command for each file
        python src/invoice_extract/main.py "$file" --model="$model"
    fi
done
