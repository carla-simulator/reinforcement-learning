Reinforcement Learning in CARLA
===============

We release a trained RL agent from the CoRL-2017 paper "CARLA: An Open Urban Driving Simulator". This is only the inference code, the training code is not released yet.

The agent was trained with the asynchronous advantage actor-critic (A3C) algorithm by V. Mnih et al. (2016). We build on this open-source Chainer implementation: https://github.com/muupan/async-rl .

Dependencies
-------
Tested with:

- CARLA 0.8.2
- python 3.6
- chainer 1.24.0
- cached-property 1.4.2
- PIL 5.1.0
- opencv 3.3.1
- h5py 2.7.1

In Anaconda, you can create and activate an environment with installed dependencies (except for CARLA) by running:
```
conda create -n carla_rl python=3.6 chainer=1.24.0 cached-property=1.4.2 pillow=5.1.0 opencv=3.3.1 h5py=2.7.1
source activate carla_rl
```

To start evaluation on the CoRL-2017 benchmark:
-------
- Start a CARLA server on town TownXX (Town01 or Town02) and port PORT (this is to be executed in the CARLA server folder):
```
./CarlaUE4.sh /Game/Maps/TownXX -carla-server -benchmark -fps=10 -windowed -ResX=800 -ResY=600 -carla-world-port=PORT
```
- Make sure CARLA client is in your python path, e.g. by running:
```
export PYTHONPATH=/path/to/CARLA/PythonClient:$PYTHONPATH
```
- Run the evaluation:
```
python run_RL.py --city-name TownXX --port PORT --corl-2017
```
The results will be stored in \_benchmarks_results.

Paper
-----

If you use this code in your research, please cite our CoRL 2017 paper:
```
@inproceedings{Dosovitskiy17,
  title = { {CARLA}: {An} Open Urban Driving Simulator},
  author = {Alexey Dosovitskiy and German Ros and Felipe Codevilla and Antonio Lopez and Vladlen Koltun},
  booktitle = {Proceedings of the 1st Annual Conference on Robot Learning},
  pages = {1--16},
  year = {2017}
}
```
