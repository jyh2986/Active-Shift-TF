from __future__ import absolute_import
import tensorflow as tf
import os.path as osp
from tensorflow.python.framework import ops


filename = osp.join(osp.dirname(__file__), 'active_shift2d.so')
_active_shift2d_module = tf.load_op_library(filename)
active_shift2d_op = _active_shift2d_module.active_shift2d_op
active_shift2d_grad_op = _active_shift2d_module.active_shift2d_backprop_op


@ops.RegisterGradient("ActiveShift2DOp")
def _active_shift2d_grad(op, grad):
  """The gradients for `active_shift2d`.
  Args:
    op: The `active_shift2d` `Operation` that we are differentiating, which we can use
      to find the inputs and outputs of the original op.
    grad: Gradient with respect to the output of the `active_shift2d` op.
  Returns:
    Gradients with respect to the input of `active_shift2d`.
  """
  data = op.inputs[0]
  shift = op.inputs[1]
  
  strides = op.get_attr('strides')
  paddings = op.get_attr('paddings')
  data_format = op.get_attr('data_format')
  normalize = op.get_attr('normalize')

  # compute gradient
  data_grad = active_shift2d_grad_op(data, shift, grad, strides, paddings, normalize, data_format)

  return data_grad  # List of one Tensor, since we have one input
