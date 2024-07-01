#!/bin/bash

python3 run_qd_grasp.py -r panda_2f -o ycb_mug -a contact_me_scs -nbr 20000 -ll -ii
python3 run_qd_grasp.py -r bh280 -o ycb_spatula -a contact_me_scs -nbr 20000 -ll -ii
python3 run_qd_grasp.py -r allegro -o ycb_power_drill -a contact_me_scs -nbr 20000 -ll -ii
python3 run_qd_grasp.py -r shadow -o ycb_rubiks_cube -a contact_me_scs -nbr 20000 -ll -ii

