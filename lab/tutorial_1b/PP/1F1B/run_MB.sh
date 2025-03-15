#!/bin/bash
cd "$(dirname "$0")" || return
START_TIME=$SECONDS

for ((i=0; i<3; i=i+1))
do
    touch "out$i.txt"
    (sleep 1; python -u "intro_PP_1F1B_MB.py" $i>"out_MB$i.txt") &
done

wait
echo "Elapsed time (s): $((SECONDS - START_TIME))"
