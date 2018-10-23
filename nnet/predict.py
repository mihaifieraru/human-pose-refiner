import tensorflow as tf

from dataset.pose_dataset import Batch
from nnet.pose_net import PoseNet
import numpy as np
from pprint import pprint

def setup_pose_prediction(cfg):
    tf.reset_default_graph()
    batch = {}
    batch[Batch.inputs] = tf.placeholder(tf.float32, shape=[1, None, None, 3])
    batch[Batch.prev_scmap] = tf.placeholder(tf.float32, shape=[1, None, None, cfg.num_joints])

    outputs = PoseNet(cfg).test(batch)

    restorer = tf.train.Saver()

    sess = tf.Session()

    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())

    # Restore variables from disk.
    restorer.restore(sess, cfg.init_weights)

    return sess, batch, outputs


def extract_cnn_output(outputs_np, cfg):
    scmap = outputs_np['part_prob']
    scmap = np.squeeze(scmap)
    locref = None
    if cfg.location_refinement:
        locref = np.squeeze(outputs_np['locref'])
        shape = locref.shape
        # print(shape)
        locref = np.reshape(locref, (shape[0], shape[1], -1, 2))
        locref *= cfg.locref_stdev
    return scmap, locref


def argmax_pose_predict(scmap, offmat, stride):
    """Combine scoremap and offsets to the final pose."""
    num_joints = scmap.shape[2]
    pose = []
    for joint_idx in range(num_joints):
        maxloc = np.unravel_index(np.argmax(scmap[:, :, joint_idx]),
                                  scmap[:, :, joint_idx].shape)
        offset = np.array(offmat[maxloc][joint_idx])[::-1] if offmat is not None else 0
        pose_f8 = (np.array(maxloc).astype('float') * stride + 0.5 * stride +
                  offset)
        pose.append(np.hstack((pose_f8[::-1],
                               [scmap[maxloc][joint_idx]])))
    return pose

def thresh_pose_confidence(cfg, pose_with_confidence):
    new_pose = []
    for joint_id in range(cfg.num_joints):
        if pose_with_confidence[joint_id][2] > cfg.joint_thresh_refine:
            new_pose.append([joint_id, pose_with_confidence[joint_id][0], pose_with_confidence[joint_id][1], pose_with_confidence[joint_id][2]])
    return new_pose

def flip_scmap(cfg, scmap_input):
    scmap_input_flip = np.expand_dims(np.fliplr(np.squeeze(np.copy(scmap_input))), axis=0)
    for joint_pair in cfg.all_joints:
        if len(joint_pair) == 2:
            tmp = np.copy(scmap_input_flip[:, :, :, joint_pair[0]])
            scmap_input_flip[:, :, :, joint_pair[0]] = np.copy(scmap_input_flip[:, :, :, joint_pair[1]])
            scmap_input_flip[:, :, :, joint_pair[1]] = tmp
    return scmap_input_flip


def fw_pass(cfg, sess, outputs, batch_inputs, rgb_input, scmap_input):
    outputs_np = sess.run(outputs, feed_dict={batch_inputs[Batch.inputs]: rgb_input, batch_inputs[Batch.prev_scmap]: scmap_input})
    if cfg.runtime_flip:    
        rgb_input_flip = np.expand_dims(np.fliplr(np.squeeze(rgb_input)), axis=0)
        scmap_input_flip = flip_scmap(cfg, scmap_input)
        outputs_np_flip = sess.run(outputs, feed_dict={batch_inputs[Batch.inputs]: rgb_input_flip, batch_inputs[Batch.prev_scmap]: scmap_input_flip})
        scmap_output_flip = flip_scmap(cfg, outputs_np_flip['part_prob'])
        outputs_np['part_prob'] = (outputs_np['part_prob'] + scmap_output_flip) / 2.0
 
    return outputs_np


def run_pose_prediction(cfg, input_data, sess, batch_inputs, outputs):
    rgb_input = np.expand_dims(input_data[0], axis=0)
    scmap_input = np.expand_dims(input_data[1], axis=0)
    
    outputs_np = fw_pass(cfg, sess, outputs, batch_inputs, rgb_input, scmap_input)
    
    scmap, locref = extract_cnn_output(outputs_np, cfg)
    pose_with_confidence = argmax_pose_predict(scmap, locref, cfg.stride)
    pose_with_confidence = thresh_pose_confidence(cfg, pose_with_confidence)
    
    return pose_with_confidence
