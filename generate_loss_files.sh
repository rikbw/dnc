#!/bin/bash
for i in {1..10}
do
    echo "Training for iteration ${i}"
    python train.py --error_file_name=error_rom_read_rounded${i} --checkpoint_interval=-1 --report_interval=250 --num_training_iterations=1000
done