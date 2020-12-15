# Framework for Efficient Robotic Manipulation
FERM is a framework that enables robots to learn tasks within an hour of real time training. Project Page: [https://sites.google.com/view/efficient-robotic-manipulation](https://sites.google.com/view/efficient-robotic-manipulation).

The project and this codebase are joint work by Albert Zhan*, Ruihan (Philip) Zhao*, Lerrel Pinto, Pieter Abbeel, Misha Laskin. The implementation is based off of [RAD](https://github.com/mlaskin/rad).

## Getting Started

Create a conda environment, and install the necessary packages.

```
conda create -n ferm python=3.7
pip install -r requirements.txt
```

## Running Experiments

Sample scripts are included in the ```scripts``` folder. This includes training, evaluating, as well as behavior cloning baselines. To launch the experiments, navigate the project root folder and run

```
./scripts/script_name.sh
```

### Robotic Experiments

To run robotic experiments, create your [gym environment](https://github.com/openai/gym) interface with your robotic setup, and substitute the ```--domain_name``` flag with your registered environment name.

### Using Demonstrations

#### Real world demonstrations

To use demonstrations, save the ```(obs, next_obs, actions, rewards, not_dones)``` demonstration tuple (a tuple of X length lists) into ```0_X.pt```, where ```X``` is the number of entries saved. Include the ```--replay_buffer_load_dir=work_directory_path/0_X.pt```

#### Sim demonstrations

Our sim experiments use large amounts of demonstrations, which are generated on the fly through an expert policy that uses state input. Include the tags ```--demo_model_dir=path_to_expert --demo_model_step=X```, where the expert policy is saved as ```path_to_expert/model/actor_X.pt``` and ```path_to_expert/model/critic_X.pt```.
