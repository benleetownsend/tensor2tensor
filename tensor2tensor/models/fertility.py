from tensor2tensor.layers import common_attention
from tensor2tensor.layers import common_layers
from tensor2tensor.models.transformer import transformer_prepare_encoder
import tensorflow as tf
import numpy as np


class FertilityModel:
  def __init__(self, max_fertility, hparams, reinforce=True):
    self.reinforce = reinforce
    self.max_fertility = max_fertility
    self.hparams = hparams
    self.exponential_adv_reward = None
    self.q_vars = []
    if reinforce:
      with tf.variable_scope(tf.VariableScope(True)):
        with tf.variable_scope("body/model/parallel_0/body/", reuse=False):
          self.exponential_adv_reward = tf.Variable(0.0, name="adv_reward", trainable=False)

  def softmax(self, x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

  def sample(self, rewards):
    return common_layers.sample_with_temperature(rewards, 1.0)

  def pyfunc_remap(self, inputs, sampled_fertilites):
    outputs = np.zeros_like(inputs)
    inputs_squeezed = np.squeeze(inputs, axis=(2, 3))
    fertilities = np.zeros_like(inputs_squeezed)
    for batch_id, sentence in enumerate(inputs_squeezed):
      current_idx = 0
      for token_idx, token in enumerate(sentence):
        if batch_id == 0:
          fertility = sampled_fertilites[batch_id, token_idx]
        outputs[batch_id, current_idx:current_idx + fertility, 0, 0] = token
        current_idx += fertility
        fertilities[batch_id, token_idx] = fertility
    return outputs, fertilities

  def fertilise(self, encoder_output, inputs):
    expected_rewards = self._qnetwork(tf.stop_gradient(encoder_output))
    expected_rewards = tf.squeeze(expected_rewards, 2)
    sampled_fertilites = self.sample(expected_rewards)
    outputs, actions = tf.py_func(self.pyfunc_remap, [inputs, sampled_fertilites], [tf.int32, tf.int32])
    return outputs, (actions, expected_rewards)

  def reinforce_op(self, actions_expected_reward, reward):
    reward = tf.stop_gradient(reward)
    actions, expected_rewards = actions_expected_reward
    # reward : [batch_size]
    # actions: [batch_size, seq_length]
    # actions_expected_reward: [batch_size, seq_length, max_fertility]

    if self.reinforce:
      # REINFORCE policy grads
      adverage_op = tf.assign(self.exponential_adv_reward,
                              0.99 * self.exponential_adv_reward + tf.reduce_mean(reward) * 0.01)
      with tf.control_dependencies([adverage_op]):
        normalised_rewards = reward - self.exponential_adv_reward
      loss_ish = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=actions, logits=expected_rewards)
      loss = tf.reduce_mean(loss_ish * normalised_rewards)

    else:
      # Q_learning
      mask = tf.one_hot(actions, self.max_fertility, axis=2)
      action_reward = tf.reduce_sum(tf.reduce_sum(expected_rewards * mask, axis=2),
                                    axis=1)  # [batch_size] #action reward
      error = tf.abs(reward - action_reward)
      clipped_error = tf.clip_by_value(error, 0.0, 1.0)
      linear_error = 2 * (error - clipped_error)
      loss = tf.reduce_mean(tf.square(clipped_error) + linear_error)

    l2_reg = weight_decay(0.0001, self.q_vars, skip_biases=False)
    return loss + l2_reg

  def _qnetwork(self, encoder_output, reuse=False):
    num_filter = 100
    with tf.variable_scope("Q_network", reuse=reuse):
      # batch_size, sequence_length, hidden_size
      encoder_output = transformer_layer(encoder_output, self.hparams)
      encoder_output = tf.expand_dims(encoder_output, -1)

      filter_shape = [1, self.hparams.hidden_size, 1, num_filter]
      W = tf.get_variable("W_1", shape=filter_shape)
      b = tf.get_variable("b_1", shape=[num_filter])
      conv = tf.nn.conv2d(
        encoder_output,
        W,
        strides=[1, 1, 1, 1],
        padding="VALID",
        name="conv")
      h = tf.nn.relu(tf.nn.bias_add(conv, b))
      self.q_vars += [W, b]

      # batch_size, sequence_length, num_filter
      filter_shape = [1, 1, num_filter, self.max_fertility]
      W = tf.get_variable("W_2", shape=filter_shape)
      b = tf.get_variable("b_2", shape=[self.max_fertility])
      conv = tf.nn.conv2d(
        h,
        W,
        strides=[1, 1, 1, 1],
        padding="VALID",
        name="conv")
      self.q_vars += [W, b]
      return tf.nn.bias_add(conv, b)


def weight_decay(decay_rate, var_list, skip_biases=True):
  if not decay_rate:
    return 0.

  tf.logging.info("Applying weight decay, decay_rate: %0.5f", decay_rate)

  weight_decays = []
  for v in var_list:
    # Weight decay.
    # This is a heuristic way to detect biases that works for main tf.layers.
    is_bias = len(v.shape.as_list()) == 1 and v.name.endswith("bias:0")
    if not (skip_biases and is_bias):
      with tf.device(v.device):
        v_loss = tf.nn.l2_loss(v)
        weight_decays.append(v_loss)

  return tf.add_n(weight_decays) * decay_rate


def transformer_layer(inputs, hparams):
  inputs = common_layers.flatten4d3d(inputs)
  x, encoder_self_attention_bias, _ = transformer_prepare_encoder(inputs, 0, hparams)
  with tf.variable_scope("fertility_attn"):
    with tf.variable_scope("self_attention"):
      y = common_attention.multihead_attention(
        common_layers.layer_preprocess(x, hparams),
        None,
        encoder_self_attention_bias,
        hparams.attention_key_channels or hparams.hidden_size,
        hparams.attention_value_channels or hparams.hidden_size,
        hparams.hidden_size,
        hparams.num_heads,
        hparams.attention_dropout,
        attention_type=hparams.self_attention_type,
        max_relative_position=hparams.max_relative_position)
      x = common_layers.layer_postprocess(x, y, hparams)
    return common_layers.layer_preprocess(x, hparams)
