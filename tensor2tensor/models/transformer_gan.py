from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Dependency imports

from tensor2tensor.layers import common_hparams
from tensor2tensor.layers import common_layers
from tensor2tensor.utils import registry
from tensor2tensor.models.transformer import Transformer, transformer_encoder, transformer_prepare_encoder

import tensorflow as tf


def reverse_grad(x):
    return tf.stop_gradient(2 * x) - x


def discriminator(embedded_trans, embedded_context, target_space_id, hparams, reuse=False):
    """Initalizes discriminator layers."""

    def model_fn_body(inputs):
        inputs = common_layers.flatten4d3d(inputs)

        (encoder_input, encoder_self_attention_bias, _) = (
            transformer_prepare_encoder(inputs, target_space_id, hparams))

        encoder_input = tf.nn.dropout(encoder_input,
                                      1.0 - hparams.layer_prepostprocess_dropout)
        encoder_output = transformer_encoder(encoder_input,
                                             encoder_self_attention_bias, hparams)
        return encoder_output

    with tf.variable_scope("discriminator", reuse=reuse):
        with tf.variable_scope("context", reuse=reuse):
            context_embed = tf.contrib.layers.layer_norm(model_fn_body(embedded_context))
        with tf.variable_scope("response", reuse=reuse):
            trans_embed = tf.contrib.layers.layer_norm(model_fn_body(embedded_trans))
    discrim_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='discriminator')
    clip = [v.assign(tf.clip_by_value(v, -0.01, 0.01)) for v in discrim_vars]

    with tf.control_dependencies(clip):
        return tf.reduce_mean(tf.matmul(context_embed, trans_embed, transpose_b=True))


@registry.register_model
class TransformerGAN(Transformer):
    def model_fn_body(self, features):
        """TransformerGAN main model_fn."""
        outputs = super(TransformerGAN, self).model_fn_body(features)
        # train = self._hparams.mode == tf.estimator.ModeKeys.TRAIN

        g_sample = outputs

        self.hparams.epsilon = 1e-5

        d_real = discriminator(tf.stop_gradient(features["targets"]), tf.stop_gradient(features["inputs"]), features["target_space_id"],
                               self._hparams)
        d_fake = discriminator(reverse_grad(g_sample), tf.stop_gradient(features["inputs"]), features["target_space_id"],
                               self._hparams, reuse=True)

        d_real = tf.Print(d_real, [d_real, d_fake])

        d_loss = tf.reduce_mean(d_fake - d_real)
        g_loss = tf.reduce_mean(-d_fake)

        losses = dict()
        losses["discriminator"] = d_loss
        losses["generator"] = g_loss

        return outputs, losses
