# QD-Grasp 6DoF


# Quality Diversity for Generating 6DoF Grasps in Robotics (QD-Grasp-6DoF)

![](assets/panda_cracker_low_qual.gif)![](assets/barrett_spatula_low_qual.gif)

![](assets/allegro_cube_low_qual.gif)![](assets/shadow_tennis_low_qual.gif)

![](assets/exp_réelles_allegro_pd_1_low_qual.gif) ![](assets/exp_réelles_allegro_pd_2_low_qual.gif)

## About
This code allows the generation of repertoires of diverse and high-performing 6DoF grasp poses with Quality-Diversity methods.

Associated paper: *Speeding up 6-DoF Grasp Sampling with Quality-Diversity, Huber, J., Hélénon, F., Kappel, M., Chelly, E., Ben Amar, F., Doncieux, S. (2024), IROS 2024* ([draft version](https://arxiv.org/abs/2403.06173))

Visit the **project webpage** for more details: [https://qdgrasp.github.io/](https://qdgrasp.github.io/)


### Supported platforms

* Franka Emika Panda Gripper (*panda_2f*)
* Robotiq Gripper 85 (*robotiq_2f*)
* Barrett Hand 280 (*bh280*) 
* Allegro Hand (*allegro*)
* Shadow Hand (*shadow*)


## Before starting

### Requirement

* python 3.10

### Recommandations

* larger number of cpu cores => faster exploration



## Install
Create a virtual env, and source it:
```
python3.10 -m venv qdg
source qdg/bin/activate
```
Launch the installer:
```
./launchers/installer.sh
```



## Launch
Always make sure to be in the virtual env:
```
source qdg/bin/activate
```

Then run one of the following examples:

### Grasps generation

Display mode, to visualize each evaluation in a sequential run: 
```
python run_qd_grasp.py -a contact_me_scs -r panda_2f -nbr 2000 -o ycb_mug -d
```

Longer run with parallelization:
```
python run_qd_grasp.py -a contact_me_scs -r panda_2f -nbr 20000 -o ycb_mug -ll
```

With Domain-Randomization:
```
python run_qd_grasp.py -a contact_me_scs -r panda_2f -nbr 50000 -o ycb_mug -ll -drf
```



### Visualizing output

To replay successful grasps from a completed run:
```
python visualization/replay_grasps.py -r path_to_run_folder/
```
Shuffle grasps and replay only robust grasps from a completed run:
```
python visualization/replay_grasps.py -r path_to_run_folder/ -si -rb
```
Visualise the success archive as fitness heatmap:
```
python visualization/plot_success_archive_heatmap.py -r path_to_run_folder/
```





