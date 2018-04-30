import tensorflow as tf
import numpy as np
from six.moves import xrange  # pylint: disable=redefined-builtin
from tensor2tensor.layers import common_layers
from tensor2tensor.utils.gradient_manip import reverse_and_reduce_gradient
from nltk.translate.gleu_score import sentence_gleu


def clip_op(variable_name):
  d_vars = [var for var in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES) if variable_name in var.name]
  clip_ops = []
  for var in d_vars:
    clip_bounds = [-.002, .002]
    clip_ops.append(
      tf.assign(
        var,
        tf.clip_by_value(var, clip_bounds[0], clip_bounds[1])
      )
    )
  return clip_ops


def linear(input_, output_size, scope=None):
  shape = input_.get_shape().as_list()

  if len(shape) != 2:
    raise ValueError("Linear is expecting 2D arguments: %s" % str(shape))
  if not shape[1]:
    raise ValueError("Linear expects shape[1] of arguments: %s" % str(shape))
  input_size = shape[1]
  # Now the computation.
  with tf.variable_scope(scope or "SimpleLinear"):
    matrix = tf.get_variable("Matrix", [output_size, input_size], dtype=input_.dtype)
    bias_term = tf.get_variable("Bias", [output_size], dtype=input_.dtype)
    return tf.matmul(input_, tf.transpose(matrix)) + bias_term


def mlp(input_, size, num_layers=1, f=tf.nn.relu, scope='Highway'):
  output = input_
  with tf.variable_scope(scope):
    for idx in xrange(num_layers):
      output = f(linear(input_, size, scope='highway_lin_%d' % idx))
      # Not actually a highway network but named this way for checkpoint compatability reasons.
      # TODO(Ben) When the old checkpoints arnt needed anymore, change the scope names
  return output


def critic(embedded_trans, embedded_context, hparams, usage, reuse=False):
  """
  Usage in ["real", "fake", "gp"]
  """
  leaky_relu = lambda x: tf.maximum(x, hparams.relu_leak * x)

  if embedded_context is not None:
    embedded_context = tf.stop_gradient(embedded_context)

  if usage == "real":
    embedded_trans = tf.stop_gradient(embedded_trans)
  elif usage == "fake":
    embedded_trans = reverse_and_reduce_gradient(embedded_trans, hparams)
  elif usage == "gp":
    embedded_trans = embedded_trans
  else:
    raise KeyError("usage not in real, fake or gp")

  filter_sizes = [1, 2, 3, 4, 6, 8, 10]
  num_filters = [500, 500, 500, 500, 200, 200, 100]
  h0, i0 = common_layers.pad_to_same_length(embedded_trans, embedded_context,
                                            final_length_divisible_by=max(filter_sizes))

  with tf.variable_scope("discriminator", reuse=reuse):
    # TODO(BEN) change "discriminator" to critic once old checkpoints not required.
    h0 = tf.expand_dims(tf.squeeze(h0, -2), -1)
    i0 = tf.expand_dims(tf.squeeze(i0, -2), -1)

    pooled_outputs = []
    if hparams.max_length == 0:
      raise Exception("Max length must be set")
    for embedded, data_name in zip([h0, i0], ["trans", "context"]):
      for filter_size, num_filter in zip(filter_sizes, num_filters):
        with tf.variable_scope("conv-maxpool-%s-%s" % (filter_size, data_name), reuse=reuse):
          # Convolution Layer
          filter_shape = [filter_size, hparams.hidden_size, 1, num_filter]
          W = tf.get_variable("W", shape=filter_shape)
          b = tf.get_variable("b", shape=[num_filter])
          conv = tf.nn.conv2d(
            embedded,
            W,
            strides=[1, 1, 1, 1],
            padding="VALID",
            name="conv")
          h = leaky_relu(tf.nn.bias_add(conv, b))

          pooled = tf.reduce_mean(h, axis=1, keep_dims=True)
          pooled_outputs.append(pooled)
    num_filters_total = sum(num_filters) * 2
    h_pool = tf.concat(pooled_outputs, 3)
    h_pool_flat = tf.reshape(h_pool, [-1, num_filters_total])

    with tf.variable_scope("highway", reuse=reuse):
      h_highway = mlp(h_pool_flat, h_pool_flat.get_shape()[1], num_layers=1, f=leaky_relu)

    with tf.variable_scope("dropout", reuse=reuse):
      h_drop = tf.nn.dropout(h_highway, 1.0 - hparams.discrim_dropout)

    with tf.variable_scope("output", reuse=reuse):
      W = tf.get_variable("W", shape=[num_filters_total, 1])
      b = tf.get_variable("b", shape=[1])
      score = tf.nn.xw_plus_b(h_drop, W, b, name="scores")
    return score


def batch_gleu(targets, model_output):
  return np.array([sentence_gleu([targ], out) for targ, out in zip(targets, model_output)], dtype=np.float32)


def decomposed_gleu(target, model_output, action):
  target = np.array(target)
  model_output = np.array(model_output)
  base = sentence_gleu([target], model_output, min_len=2, max_len=2)
  deltas = np.zeros_like(model_output, dtype=np.float32)
  for i, _ in enumerate(model_output):
    deltas[i] = base - sentence_gleu([target], np.concatenate((model_output[:i], model_output[i + 1:])), min_len=2,
                                     max_len=2)
  factored_gleu = base * np.exp(deltas) / sum(np.exp(deltas))
  index = 0
  factored_sumed_gleu = np.zeros_like(action, dtype=np.float32)
  for i, act in enumerate(action):
    factored_sumed_gleu[i] = np.sum(factored_gleu[index:index + act])
    index += act
  return factored_sumed_gleu


def batch_decomposed_gleu(targets, model_output, actions):
  return np.array([decomposed_gleu(targ, out, act) for targ, out, act in zip(targets, model_output, actions)],
                  dtype=np.float32)


def tf_decomposed_gleu(targets, model_output, actions):
  # both : [batch, seq_length, 1, 1]
  targets = tf.squeeze(targets, (2, 3))
  model_output = tf.squeeze(model_output, (2, 3))
  return tf.py_func(batch_decomposed_gleu, [targets, model_output, actions], tf.float32)


def tf_gleu(targets, model_output):
  # both : [batch, seq_length, 1, 1]
  targets = tf.squeeze(targets, (2, 3))
  model_output = tf.squeeze(model_output, (2, 3))
  return tf.py_func(batch_gleu, [targets, model_output], tf.float32)


def gen_transpose_embeddings(raw_targets, targets, embeddings, hparams):
  trans_embeddings = tf.get_variable("trans_embed", shape=embeddings.get_shape().as_list())

  targets_transpose_embed = tf.gather(trans_embeddings, raw_targets)
  targets_transpose_embed = tf.reshape(targets_transpose_embed, tf.shape(targets))

  projected_truth = tf.matmul(tf.reshape(targets_transpose_embed, [-1, hparams.hidden_size]),
                              tf.stop_gradient(embeddings), transpose_b=True)

  trans_embed_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
    labels=tf.reshape(raw_targets, [-1]),
    logits=projected_truth,
    name="trans_embed_loss"
  )
  targets_transpose_embed += tf.random_normal(tf.shape(targets_transpose_embed), stddev=0.1)

  return trans_embed_loss, targets_transpose_embed


def pyfunc_remap(inputs):
  """ This generates a random plan for the model. """
  outputs = np.zeros_like(inputs)
  inputs_squeezed = np.squeeze(inputs, axis=(2, 3))
  sampled_fertilites = np.random.randint(0, 4, inputs_squeezed.shape)
  fertilities = np.zeros_like(inputs_squeezed)
  for batch_id, sentence in enumerate(inputs_squeezed):
    current_idx = 0
    for token_idx, token in enumerate(sentence):
      if batch_id == 0:
        fertility = sampled_fertilites[batch_id, token_idx]
      outputs[batch_id, current_idx:current_idx + fertility, 0, 0] = token
      current_idx += fertility
      fertilities[batch_id, token_idx] = fertility
  return outputs


def randomly_fertilise(inputs):
  """ Tensorflow wrapper for pyfunc_remap """
  outputs = tf.py_func(pyfunc_remap, [inputs], [tf.int32])
  outputs = tf.reshape(outputs, tf.shape(inputs))
  return outputs
