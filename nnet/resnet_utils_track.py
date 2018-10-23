# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Contains building blocks for various versions of Residual Networks.

Residual networks (ResNets) were proposed in:
  Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
  Deep Residual Learning for Image Recognition. arXiv:1512.03385, 2015

More variants were introduced in:
  Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
  Identity Mappings in Deep Residual Networks. arXiv: 1603.05027, 2016

We can obtain different ResNet variants by changing the network depth, width,
and form of residual unit. This module implements the infrastructure for
building them. Concrete ResNet units and full ResNet networks are implemented in
the accompanying resnet_v1.py and resnet_v2.py modules.

Compared to https://github.com/KaimingHe/deep-residual-networks, in the current
implementation we subsample the output activations in the last residual unit of
each block, instead of subsampling the input activations in the first residual
unit of each block. The two implementations give identical results but our
implementation is more memory efficient.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


from tensorflow.contrib import layers as layers_lib
from tensorflow.python.ops import array_ops
import tensorflow as tf




def conv2d_same_track(inputs, num_outputs, kernel_size, stride, rate=1, scope=None):
  """Strided 2-D convolution with 'SAME' padding.

  When stride > 1, then we do explicit zero-padding, followed by conv2d with
  'VALID' padding.

  Note that

     net = conv2d_same(inputs, num_outputs, 3, stride=stride)

  is equivalent to

     net = tf.contrib.layers.conv2d(inputs, num_outputs, 3, stride=1,
     padding='SAME')
     net = subsample(net, factor=stride)

  whereas

     net = tf.contrib.layers.conv2d(inputs, num_outputs, 3, stride=stride,
     padding='SAME')

  is different when the input's height or width is even, which is why we add the
  current function. For more details, see ResnetUtilsTest.testConv2DSameEven().

  Args:
    inputs: A 4-D tensor of size [batch, height_in, width_in, channels].
    num_outputs: An integer, the number of output filters.
    kernel_size: An int with the kernel_size of the filters.
    stride: An integer, the output stride.
    rate: An integer, rate for atrous convolution.
    scope: Scope.

  Returns:
    output: A 4-D tensor of size [batch, height_out, width_out, channels] with
      the convolution output.
  """
  input_conv_all = []
  scopes = [scope, scope + '_extra']
  for id in range(len(inputs)):
      input = inputs[id]
      if stride == 1:
        input_conv = layers_lib.conv2d(
            input,
            num_outputs,
            kernel_size,
            stride=1,
            rate=rate,
            padding='SAME',
            activation_fn=None,
            normalizer_fn=None,
            biases_initializer=None,
            # weights_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
            scope=scopes[id])
      else:
        kernel_size_effective = kernel_size + (kernel_size - 1) * (rate - 1)
        pad_total = kernel_size_effective - 1
        pad_beg = pad_total // 2
        pad_end = pad_total - pad_beg
        input = array_ops.pad(
            input, [[0, 0], [pad_beg, pad_end], [pad_beg, pad_end], [0, 0]])
        input_conv = layers_lib.conv2d(
            input,
            num_outputs,
            kernel_size,
            stride=stride,
            rate=rate,
            padding='VALID',
            activation_fn=None,
            normalizer_fn=None,
            biases_initializer=None,
            # weights_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
            scope=scopes[id])

      input_conv_all.append(input_conv)

  return tf.nn.relu(layers_lib.batch_norm(sum(input_conv_all)))