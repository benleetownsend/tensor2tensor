import tensorflow as tf
from tensor2tensor.layers import common_layers


def reverse_grad(x):
  return tf.stop_gradient(2 * x) - x


def reverse_and_reduce_gradient(x, hparams=None):
  if hparams is None:
    grad_mul = 0.01
    step_interval = None
    warmup = 0

  else:
    grad_mul = hparams.discrim_grad_mul
    step_interval = hparams.step_interval
    warmup = hparams.warmup_steps

  if step_interval is not None:
    global_step = tf.train.get_or_create_global_step()
    gating_multiplier = tf.to_float(
      tf.logical_and(tf.equal(tf.mod(global_step, step_interval), 0), tf.greater(global_step, warmup)))

    gan_step = tf.maximum(tf.to_float(tf.train.get_global_step()) - warmup, 0)
    progress = tf.minimum(gan_step / float(warmup * 2), 1.0)
    decay_multiplier = progress
    grad_mul *= gating_multiplier
    grad_mul *= decay_multiplier

    tf.summary.scalar("gen_grad_mul", grad_mul)

  return tf.stop_gradient(x + grad_mul * x) - grad_mul * x


def decay_gradient(outputs, decay_period, final_val=1.00, summarize=True):
  masking = common_layers.inverse_lin_decay(decay_period)
  masking = tf.minimum(tf.maximum(masking, 0.0), final_val)
  if summarize:
    tf.summary.scalar("loss_mask", masking)
  return tf.stop_gradient(masking * outputs) + (1.0 - masking) * outputs
