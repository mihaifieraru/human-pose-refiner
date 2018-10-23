import re

import tensorflow as tf
import tensorflow.contrib.slim as slim
# from tensorflow.contrib.slim.nets import resnet_v1
import nnet.resnet_v1_track as resnet_v1

import numpy as np

from dataset.pose_dataset import Batch
from nnet import losses


net_funcs = {'resnet_50': resnet_v1.resnet_v1_50,
             'resnet_101': resnet_v1.resnet_v1_101}


def prediction_layer(cfg, input, name, num_outputs):
    with slim.arg_scope([slim.conv2d, slim.conv2d_transpose], padding='SAME',
                        activation_fn=None, normalizer_fn=None,
                        weights_regularizer=slim.l2_regularizer(cfg.weight_decay)):
        with tf.variable_scope(name):
            pred = slim.conv2d_transpose(input, num_outputs,
                                         kernel_size=[3, 3], stride=2,
                                         scope='block4')
            return pred


def get_batch_spec(cfg):
    num_joints = cfg.num_joints
    batch_size = cfg.batch_size
    batch_spec = {
        Batch.inputs: [batch_size, None, None, 3],
        Batch.part_score_targets: [batch_size, None, None, num_joints],
        Batch.part_score_weights: [batch_size, None, None, num_joints]
    }
    if cfg.location_refinement:
        batch_spec[Batch.locref_targets] = [batch_size, None, None, num_joints * 2]
        batch_spec[Batch.locref_mask] = [batch_size, None, None, num_joints * 2]
    batch_spec[Batch.prev_scmap] = [batch_size, None, None, num_joints]
    return batch_spec


class PoseNet:
    def __init__(self, cfg):
        self.cfg = cfg

    def extract_features(self, net_inputs):
        net_inputs_normalized = []
        net_fun = net_funcs[self.cfg.net_type]

        mean = tf.constant(self.cfg.mean_pixel[0:3],
                           dtype=tf.float32, shape=[1, 1, 1, 3], name='img_mean')
        std = tf.constant(self.cfg.std_pixel[0:3],
                           dtype=tf.float32, shape=[1, 1, 1, 3], name='img_std')
        net_inputs_normalized.append((net_inputs[0] - mean))

        mean_extra_channels = tf.constant(self.cfg.mean_pixel[3:3+self.cfg.num_joints],
                                                dtype=tf.float32, shape=[1, 1, 1, self.cfg.num_joints], name='prev_scmap_mean')
        std_extra_channels = tf.constant(self.cfg.std_pixel[3:3 + self.cfg.num_joints],
                                              dtype=tf.float32, shape=[1, 1, 1, self.cfg.num_joints],
                                              name='prev_scmap_std')
        net_inputs_normalized.append((net_inputs[1] - mean_extra_channels))


        with slim.arg_scope(resnet_v1.resnet_arg_scope()):
            net, end_points = net_fun(net_inputs_normalized,
                                      global_pool=False, output_stride=16, is_training=False)

        return net, end_points


    def prediction_layers(self, features, end_points, reuse=None, no_interm=False, scope='pose'):
        cfg = self.cfg

        num_layers = re.findall("resnet_([0-9]*)", cfg.net_type)[0]
        layer_name = 'resnet_v1_{}'.format(num_layers) + '/block{}/unit_{}/bottleneck_v1'

        out = {}
        with tf.variable_scope(scope, reuse=reuse):
            out['part_pred'] = prediction_layer(cfg, features, 'part_pred',
                                                cfg.num_joints)
            if cfg.location_refinement:
                out['locref'] = prediction_layer(cfg, features, 'locref_pred',
                                                 cfg.num_joints * 2)
        return out

    def get_net(self, net_inputs):
        net, end_points = self.extract_features(net_inputs)
        return self.prediction_layers(net, end_points)

    def test(self, batch):
        net_inputs = [batch[Batch.inputs]]
        net_inputs.append(batch[Batch.prev_scmap])
        heads = self.get_net(net_inputs)
        return self.add_test_layers(heads)

    def add_test_layers(self, heads):
        prob = heads['part_pred']
        prob = tf.sigmoid(prob)
        outputs = {'part_prob': prob}
        if self.cfg.location_refinement:
            outputs['locref'] = heads['locref']
        return outputs

    def part_detection_loss(self, heads, batch, locref):
        cfg = self.cfg

        weigh_part_predictions = cfg.weigh_part_predictions
        part_score_weights = batch[Batch.part_score_weights] if weigh_part_predictions else 1.0

        def add_part_loss(pred_layer):
            loss_func = tf.losses.mean_squared_error if cfg.gaussian_target else tf.losses.sigmoid_cross_entropy
            return loss_func(batch[Batch.part_score_targets],
                             heads[pred_layer],
                             part_score_weights)

        loss = {}
        loss['part_loss'] = add_part_loss('part_pred')
        total_loss = loss['part_loss']

        if locref:
            locref_pred = heads['locref']
            locref_targets = batch[Batch.locref_targets]
            locref_weights = batch[Batch.locref_mask]

            loss_func = losses.huber_loss if cfg.locref_huber_loss else tf.losses.mean_squared_error
            loss['locref_loss'] = cfg.locref_loss_weight * loss_func(locref_targets, locref_pred, locref_weights)
            total_loss = total_loss + loss['locref_loss']


        # loss['total_loss'] = slim.losses.get_total_loss(add_regularization_losses=params.regularize)
        loss['total_loss'] = total_loss
        return loss

    def train(self, batch):
        cfg = self.cfg

        locref = cfg.location_refinement

        net_inputs = [batch[Batch.inputs]]
        net_inputs.append(batch[Batch.prev_scmap])
        heads = self.get_net(net_inputs)
        return self.part_detection_loss(heads, batch, locref)
