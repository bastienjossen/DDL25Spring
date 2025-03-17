#!/bin/bash
set -e  # exit immediately if any command fails
trap 'echo "An error occurred. Aborting workflow." >&2; exit 1' ERR

echo "Starting distributed training workflow with 6 processes..."
cd "$(dirname "$0")" || exit 1

START_TIME=$SECONDS

for ((i=0; i<6; i++)); do
    echo "Launching process with rank $i"
    touch "out_MP$i.txt"
    (sleep 1; python -u intro_PP_1F1B_MP_try2.py $i > "out_MP$i.txt") &
done

wait

echo "All processes completed successfully."
echo "Elapsed time (s): $((SECONDS - START_TIME))"
