from __future__ import print_function

import sys
import os

from .asyncrl import run_train_test
from .asyncrl import a3c
import numpy as np
from chainer import serializers

from carla.agent import Agent
from carla.carla_server_pb2 import Control

class Struct:
    def __init__(self, **entries):
        self.__dict__.update(entries)

class A3CAgent(Agent):
    def __init__(self, city_name, args_file='', model_file='', n_actions=0, frameskip=1):
        Agent.__init__(self)
        self.args = self.read_args(args_file)
        self.args.model = model_file
        self.n_actions = n_actions
        self.n_meas = self.compute_n_meas(self.args)
        self.args.town_traintest = city_name
        self.setup_model(self.n_actions, self.n_meas, self.args)
        self.setup_data_preprocessor(self.args)
        self.frameskip = frameskip
        self.step = 0

    def run_step(self, meas, sensory, directions, target):
        # print('Step {}'.format(self.step))
        if self.step % self.frameskip == 0:
            obs_preprocessed = self.preprocess_data(meas, sensory, directions, target)
            action_idx = self.actor.act(obs_preprocessed=obs_preprocessed)
            action = self.actions[action_idx]
            control = Control()
            if self.obs_dict['speed'] < 30.:
                control.throttle = action[0]
            elif control.throttle > 0.:
                control.throttle = 0.
            control.steer = action[1]
            self.prev_control = control
        else:
            control = self.prev_control
            # print('Repeating control')
        self.step += 1
        print(control.throttle, control.steer)
        return control

    def read_args(self, args_file):
        with open(args_file, 'r') as f:
            args_dict = eval(f.read())
        return Struct(**args_dict)

    def compute_n_meas(self, args):
        modalities = ['accel_x', 'accel_y', 'collision_car', 'collision_gen', 'collision_ped', 'game_timestamp', 'platform_timestamp', 'image', 'ori_x', 'ori_y', 'ori_z', 'player_x', 'player_y', 'intersect_otherlane', 'intersect_offroad', 'speed', 'vector_to_goal', 'distance_to_goal', 'planner_command', 'step']
        dimensionalities = {m: 1 for m in modalities}
        dimensionalities.update({'image': None, 'vector_to_goal': 2, 'planner_command': 5})
        n_meas = sum([dimensionalities[m] for m in args.meas_list])
        return n_meas

    def setup_model(self, n_actions, n_meas, args):
        self.model = run_train_test.get_model(n_actions, n_meas, args)
        serializers.load_hdf5(args.model, self.model)

        if type(self.model).__name__ == 'A3CFF':
            self.actor = a3c.A3CActor(self.model, input_preprocess=None, random_action_prob=0.)
        else:
            raise Exception('Unknown model type', type(model).__name__)

        if (not hasattr(args, 'carla_action_set')) or args.carla_action_set == '9':
            self.actions = [[0., 0.], [-1.,0.], [-0.5,0.], [0.5, 0.], [1.0, 0.], [0., -1.], [0., -0.5], [0., 0.5], [0.,1.]]
        elif args.carla_action_set == '13':
            self.actions = [[0., 0.], [-1.,0.], [-0.5,0.], [-0.25,0.], [0.25,0.], [0.5, 0.], [1.0, 0.], [0., -1.], [0., -0.5], [0., -0.25], [0., 0.25], [0., 0.5], [0.,1.]]
        else:
            raise Exception('Unknown args.carla_action_set {}'.format(args.carla_action_set))

    def setup_data_preprocessor(self, args):
        meas_coeffs_dict = {'step': [1/500.], 'vector_to_goal': [1/5000.,1/5000.], 'distance_to_goal': [1/1000.], 'speed': [1/10.], 'collision_gen': [1/500000.], \
                            'collision_ped': [1/100000.], 'collision_car': [1/500000.], 'intersect_offroad': [1.], 'intersect_otherlane': [1.], 'planner_command': [0.5,0.5,0.5,0.5,0.5]}
        meas_coeffs = np.concatenate([np.array(meas_coeffs_dict[m]) for m in args.meas_list]).astype(np.float32)
        self.input_preprocessor = run_train_test.InputPreprocessor(meas_coeffs=meas_coeffs, n_images_to_accum=args.n_images_to_accum, meas_list=args.meas_list)

    def preprocess_data(self, meas, sensory, planner_command, target):
        self.obs_dict = self.data_from_simulator_to_dict(meas, sensory, planner_command, target)
        print('Planner', self.obs_dict['planner_command'])
        obs_preprocessed = self.input_preprocessor(self.obs_dict)
        return obs_preprocessed

    def data_from_simulator_to_dict(self, measurements, sensory, planner_command, target):
        modalities = ['accel_x', 'accel_y', 'collision_car', 'collision_gen', 'collision_ped', 'game_timestamp', 'platform_timestamp', 'image', 'ori_x', 'ori_y', 'ori_z', 'player_x', 'player_y', 'intersect_otherlane', 'intersect_offroad', 'speed', 'vector_to_goal', 'distance_to_goal', 'planner_command']
        if measurements is None:
            data_dict = {m: None for m in (modalities + ['image', 'goal_pos', 'goal_ori', 'step', 'planner_command'])}
        else:
            player_measurements = measurements.player_measurements
            print(player_measurements)
            data_dict = {}
            # NOTE we convert new SI CARLA units (from 0.8.0) to old CARLA units (pre-0.8.0) since the model was trained in old CARLA
            data_dict['accel_x'] = player_measurements.acceleration.x*100.
            data_dict['accel_y'] = player_measurements.acceleration.y*100.
            data_dict['collision_car'] = player_measurements.collision_vehicles*100.
            data_dict['collision_ped'] = player_measurements.collision_pedestrians*100.
            data_dict['collision_gen'] = player_measurements.collision_other*100.
            data_dict['ori_x'] = player_measurements.transform.orientation.x
            data_dict['ori_y'] = player_measurements.transform.orientation.y
            data_dict['ori_z'] = player_measurements.transform.orientation.z
            data_dict['player_x'] = player_measurements.transform.location.x*100.
            data_dict['player_y'] = player_measurements.transform.location.y*100.
            #data_dict['player_z'] = player_measurements.transform.location.z
            data_dict['intersect_otherlane'] = player_measurements.intersection_otherlane
            data_dict['intersect_offroad'] = player_measurements.intersection_offroad
            data_dict['speed'] = player_measurements.forward_speed*3.6

            data_dict['game_timestamp'] = 0.
            data_dict['platform_timestamp'] = 0.

            print(sensory, dir(sensory['CameraRGB']))
            # print(sensory['CameraRGB'].data)
            # print(sensory['CameraRGB'].raw_data)
            data_dict['image'] = sensory['CameraRGB'].data[:,:,:-1][:,:,::-1] # get rid of A and then revert channels

            data_dict['goal_pos'] = (target.location.x*100., target.location.y*100.)
            data_dict['goal_ori'] = (target.orientation.x, target.orientation.y)
            data_dict['step'] = 0.

            pos = (data_dict['player_x'], data_dict['player_y'], 22)
            ori = (data_dict['ori_x'], data_dict['ori_y'], data_dict['ori_z'])
            goal_pos_3d = (data_dict['goal_pos'][0], data_dict['goal_pos'][1], 22)
            goal_ori_3d = (data_dict['goal_ori'][0], data_dict['goal_ori'][1], -0.001)
            data_dict['planner_command'] = planner_command

            player_pos = np.array([data_dict['player_x'], data_dict['player_y']])
            to_goal = data_dict['goal_pos'] - player_pos
            to_agents_coords_matrix = np.array([[data_dict['ori_x'], data_dict['ori_y']], [-data_dict['ori_y'], data_dict['ori_x']]])
            data_dict['vector_to_goal'] = to_agents_coords_matrix.dot(to_goal)
            data_dict['distance_to_goal'] = np.sqrt(np.sum(np.abs(to_goal)**2, keepdims=True))

            planner_command_onehot = np.zeros(5)
            if data_dict['planner_command'] is None:
                pass
            else:
                assert (data_dict['planner_command'] in [0.,2.,3.,4.,5.]), 'Got planner command {}. Expected to be one of: 0,2,3,4,5'.format(data_dict['planner_command'])
                if data_dict['planner_command'] == 0:
                    planner_command_onehot[0] = 1.
                else:
                    planner_command_onehot[int(data_dict['planner_command'])-1] = 1.
            data_dict['planner_command'] = planner_command_onehot

            if data_dict['distance_to_goal'] > 4000.:
                data_dict['distance_to_goal'] = 4000.

        for m in modalities:
            assert (m in data_dict), "data_dict should have field {}".format(m)

        return data_dict
