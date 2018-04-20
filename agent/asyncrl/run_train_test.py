import argparse
import multiprocessing as mp

import sys
sys.path.append('/home/adosovit/work/libs/opencv3/lib/python3')
sys.path=['/home/adosovit/work/libs/anaconda3/lib/python3.5/site-packages'] + sys.path

import chainer
from chainer import links as L
from chainer import functions as F
from chainer import serializers

import cv2
import numpy as np
import re
import os
import time

from . import policy, fc_net, dqn_head, a3c, weight_init


class InputPreprocessor:
    def __init__(self, meas_coeffs=None, n_images_to_accum=None, meas_list=[]):
        self.num_meas = len(meas_coeffs)
        self.meas_list = meas_list
        self.meas_coeffs = np.array(meas_coeffs, dtype=np.float32)
        self.n_images_to_accum=n_images_to_accum
        self.img_buffer = np.zeros((self.n_images_to_accum,84,84), dtype = np.uint8)
        self.step = 0

    def __call__(self, obs):
        img = obs['image']

        if img.shape[:2] == (84,84):
            resized = img
        else:
            resized = cv2.resize(img, (84, 84))

        resized = np.mean(resized,axis=2,keepdims=True)
        all_meas_list = [np.array(obs[m]) if (isinstance(obs[m], list) or isinstance(obs[m],np.ndarray)) else np.array([obs[m]]) for m in self.meas_list]
        meas = np.concatenate(all_meas_list).astype(np.float32)

        if self.step == 0:
            # at the first step, fill the buffer with identical images
            for n in range(self.n_images_to_accum):
                self.img_buffer[n] = np.squeeze(resized)
        else:
            self.img_buffer[self.step % self.n_images_to_accum] = np.squeeze(resized)
        self.step += 1

        return {'image': self.img_buffer[np.arange(self.step-1,self.step-1+self.n_images_to_accum) % self.n_images_to_accum].astype(np.float32) / 255. -0.5, \
                'meas': meas.astype(np.float32)*self.meas_coeffs,
                'raw_image': img,
                'raw_meas': meas.astype(np.float32)}

    def reset(self):
        self.img_buffer *= 0
        self.step = 0

class A3CFF(chainer.ChainList, a3c.A3CModel):

    def __init__(self, n_actions, model_type='advantage', n_meas_in=None, img_fc_layers=None, meas_fc_layers=None, joint_fc_layers=None, head_arch=None, n_input_channels=None,
                 nonlinearity_str=None, weight_init_str=None, bias_init=None):
        if head_arch == 'nature':
            self.dqn_net = dqn_head.NatureDQNHead(n_input_channels=n_input_channels, nonlinearity_str=nonlinearity_str, bias=bias_init)
        elif head_arch == 'nips':
            self.dqn_net = dqn_head.NIPSDQNHead(n_input_channels=n_input_channels, nonlinearity_str=nonlinearity_str, bias=bias_init)
        else:
            raise Exception('Unknown head architecture', head_arch)

        self.n_meas_in = n_meas_in
        self.input_meas = (meas_fc_layers > 0)
        self.n_actions = n_actions
        nch = self.dqn_net.n_output_channels

        assert joint_fc_layers >= 1, "Should have at least one joint fc layer"
        assert meas_fc_layers >= 1, "Should have at least one meas fc layer"

        self.img_fc = fc_net.FCNet([self.dqn_net.n_output_channels] + [nch]*img_fc_layers, last_nonlinearity=True, nonlinearity_str=nonlinearity_str)
        self.meas_fc = fc_net.FCNet([self.n_meas_in] + [nch]*meas_fc_layers, last_nonlinearity=True, nonlinearity_str=nonlinearity_str)
        self.joint_fc = fc_net.FCNet([2*nch] + [nch]*joint_fc_layers, last_nonlinearity=True, nonlinearity_str=nonlinearity_str)
        self.pi = policy.FCSoftmaxPolicy(nch, n_actions)
        self.v = fc_net.FCNet([nch,1], last_nonlinearity=False, nonlinearity_str=nonlinearity_str)
        super().__init__(self.dqn_net, self.img_fc, self.meas_fc, self.joint_fc, self.pi, self.v)
        weight_init.init_with_str(self, init_str=weight_init_str)

    def pi_and_v(self, img, meas=None, keep_same_state=False):
        img_feat = self.img_fc(self.dqn_net(img))
        if self.input_meas:
            meas_feat = self.meas_fc(meas)
            joint_feat = self.joint_fc(F.concat((img_feat, meas_feat), axis=1))
        else:
            raise NotImplementedError("No input measurements currently not supported")
            joint_feat = self.joint_fc(img_feat)

        return self.pi(joint_feat), self.v(joint_feat)


def get_model(n_actions, n_meas, args):
    model = A3CFF(n_actions, n_meas_in=n_meas, img_fc_layers=args.img_fc_layers, meas_fc_layers=args.meas_fc_layers, joint_fc_layers=args.joint_fc_layers, head_arch=args.img_conv_arch,
                  n_input_channels=args.n_images_to_accum, nonlinearity_str=args.nonlinearity, weight_init_str=args.weight_init, bias_init=args.bias_init)
    return model
