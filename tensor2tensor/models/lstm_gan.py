from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensor2tensor.utils import t2t_model

from tensor2tensor.layers import common_layers
from tensor2tensor.utils import registry
from tensor2tensor.models.transformer import transformer_base_single_gpu
from tensor2tensor.utils.gan_utils import clip_op, critic, gen_transpose_embeddings
from tensor2tensor.models.lstm import auto_regressive_bi_attn_lstm_s2s

import tensorflow as tf

@registry.register_model
class LstmGAN(t2t_model.T2TModel):
  def model_fn_body(self, features):
    target_modality = self._problem_hparams.target_modality

    original_targets = features["targets"]
    inputs = features.get("inputs", None)
    hparams = self._hparams

    decode_out = auto_regressive_bi_attn_lstm_s2s(inputs, hparams, target_modality,
                                                  hparams.mode == tf.estimator.ModeKeys.TRAIN,
                                                  "Generator")

    with tf.variable_scope(tf.VariableScope(True)):
      with tf.variable_scope(target_modality.name + "/shared", reuse=True):
        embeddings = target_modality._get_weights()

    trans_embed_loss, discrim_features = gen_transpose_embeddings(features["targets_raw"], original_targets, embeddings,
                                                                  hparams)

    d_real = critic(discrim_features, features["inputs"], usage="real", hparams=self._hparams)
    tf.summary.scalar("real_score", tf.reduce_mean(d_real))

    gradient_penalty = 0.0
    if hparams.ganmode == "wgan-gp":
      alpha = tf.random_uniform(shape=[tf.shape(discrim_features)[0], 1, 1, 1], minval=0., maxval=1.)

      differences = tf.stop_gradient(decode_out - discrim_features)
      interpolates = tf.stop_gradient(discrim_features) + (alpha * differences)
      gradients = tf.gradients(
        critic(interpolates, features["inputs"], usage="gp", hparams=self._hparams, reuse=True),
        [interpolates])[0]

      slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), reduction_indices=[1, 2, 3]))
      gradient_penalty = tf.reduce_mean((slopes - 1.) ** 2) * hparams.lipschitz_mult

    d_fake = critic(decode_out, features["inputs"], usage="fake", hparams=self._hparams, reuse=True)
    tf.summary.scalar("gen_score", tf.reduce_mean(d_fake))
    d_loss = tf.reduce_mean(d_fake - d_real)
    d_real_cycled = critic(tf.random_shuffle(discrim_features), features["inputs"], usage="real",
                           hparams=self._hparams, reuse=True)
    d_loss_cycle = tf.reduce_mean(d_real_cycled - d_real)

    if hparams.ganmode == "wgan":
      clip_ops = clip_op("discriminator")
      with tf.control_dependencies(clip_ops):
        d_loss = tf.identity(d_loss)

    losses = {
      "discriminator": d_loss,
      "lipschitz-penalty": gradient_penalty,
      "trans_embed_loss": trans_embed_loss,
      "semantic_reg": d_loss_cycle * 150,
    }
    if self._hparams.mode == tf.estimator.ModeKeys.PREDICT:
      return decode_out, d_fake

    return decode_out, losses

  def model_fn(self, features, **kwargs):
    features["inputs"], features["targets"] = common_layers.pad_to_same_length(features["inputs"],
                                                                               features["targets"])
    return super(LstmGAN, self).model_fn(features, **kwargs)


@registry.register_hparams
def lstm_gan_base():
  hparams = transformer_base_single_gpu()
  hparams.input_modalities = "inputs:symbol:GAN"
  hparams.target_modality = "symbol:GAN"
  hparams.batch_size = 1024
  hparams.learning_rate = 1e-5
  hparams.learning_rate_decay_scheme = "none"
  hparams.optimizer = "RMSProp"
  hparams.summarize_grads = False
  hparams.clip_grad_norm = 1000.0
  hparams.num_decoder_layers = 4
  hparams.num_encoder_layers = 4
  hparams.max_length = 600
  hparams.layer_prepostprocess_dropout = 0.1
  hparams.attention_dropout = 0.1
  hparams.relu_dropout = 0.3
  hparams.label_smoothing = 0.0
  hparams.add_hparam("ganmode", "wgan")
  hparams.add_hparam("num_compress_steps", 2)
  hparams.add_hparam("num_decode_steps", 0)
  hparams.add_hparam("discrim_grad_mul", 1e-9)
  hparams.add_hparam("step_interval", 1)
  hparams.add_hparam("warmup_steps", 22001)
  hparams.add_hparam("mle_decay_period", 22000)
  hparams.add_hparam("embed_decay_period", 22000)
  hparams.add_hparam("lipschitz_mult", 1500.0)
  hparams.add_hparam("fertility_cells", 2)
  hparams.add_hparam("z_temp", 0.05)
  hparams.add_hparam("discrim_dropout", 0.3)
  hparams.add_hparam("relu_leak", 0.2)
  hparams.add_hparam("reinforce_delta", 1e-7)
  hparams.add_hparam("embedding_file", "embeddings.pkl")
  hparams.add_hparam("fertility_filename", "ENG_FR.alignfertility_model.pkl")
  return hparams


@registry.register_hparams
def lstm_gan_german():
  hparams = lstm_gan_base()
  hparams.embedding_file = None
  hparams.z_temp = 0
  hparams.fertility_filename = None
  hparams.learning_rate = 1e-6
  hparams.reinforce_delta = 1e-1
  hparams.add_hparam("lang_model_file", None)  # "/root/code/t2t_data/lang_model_small.germ.pkl")
  hparams.add_hparam("lang_model_data", "/root/code/t2t_data/t2t_datagen/german_lang_model_data.txt")
  hparams.ganmode = "wgan-gp"
  return hparams


def decay_gradient(outputs, decay_period, final_val=1.00, summarize=True):
  masking = common_layers.inverse_lin_decay(decay_period)
  masking = tf.minimum(tf.maximum(masking, 0.0), final_val)
  if summarize:
    tf.summary.scalar("loss_mask", masking)
  return tf.stop_gradient(masking * outputs) + (1.0 - masking) * outputs
