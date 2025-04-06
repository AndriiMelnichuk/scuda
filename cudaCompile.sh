#!/bin/bash

SRC_DIR="src/main/cuda/"
RES_DIR="src/main/resources/"

mkdir -p "$RES_DIR"

for file in "$SRC_DIR"*; do
    if [[ -f "$file" ]]; then
        filename="${file##*/}"
        filename="${filename%.cu}.ptx"
        nvcc -ptx "$file" -o "$RES_DIR$filename"
    fi
done