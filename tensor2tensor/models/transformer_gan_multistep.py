from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Dependency imports

from tensor2tensor.layers import modalities
from tensor2tensor.layers import common_layers
from tensor2tensor.utils import registry
from tensor2tensor.models import transformer_vae
from tensor2tensor.models.transformer import Transformer, transformer_base_single_gpu, transformer_prepare_decoder
import tensorflow as tf
import numpy as np


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
        grad_mul *= gating_multiplier

    return tf.stop_gradient(x + grad_mul * x) - grad_mul * x


def discriminator(embedded_trans, embedded_context, hparams, usage, reuse=False):
    """
    Usage in ["real", "fake", "gp"]
    """
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

    with tf.variable_scope("discriminator", reuse=reuse):
        if embedded_context is not None:
            h0, i0 = common_layers.pad_to_same_length(
                embedded_trans, embedded_context, final_length_divisible_by=16)
            h0 = tf.concat([h0, i0], axis=-1)

        h0 = tf.layers.dense(h0, hparams.hidden_size, name="io")
        h1 = transformer_vae.compress(h0, None, False, hparams, "compress1")
        h2 = transformer_vae.compress(h1, None, False, hparams, "compress2")
        res_dense = tf.reduce_mean(h2, axis=[1, 2])
        res_single = tf.squeeze(tf.layers.dense(res_dense, 1), axis=-1)
        return res_single


class LazyPinv:
    def __init__(self, threshold=1.0):
        self.threshold = threshold
        self.cache = None
        self.state = None

    def __call__(self, inp):
        if self.state is not None:
            print("Norm = ", np.linalg.norm(self.state - inp))
        if self.state is None or np.linalg.norm(self.state - inp) > self.threshold:
            self.cache = np.linalg.pinv(inp)
            self.state = inp
        return self.cache


def pinv_and_transpose(inp):
    return tf.stop_gradient(tf.transpose(tf.py_func(LazyPinv(), [inp], tf.float32)))


@registry.register_model
class TransformerGAN(Transformer):
    def decode(self,
               decoder_input,
               encoder_output,
               encoder_decoder_attention_bias,
               decoder_self_attention_bias,
               hparams,
               cache=None,
               nonpadding=None):

        embed_dim = self._hparams.hidden_size
        noise = tf.random_uniform(shape=tf.shape(decoder_input), minval=-0.5, maxval=0.5)
        noise *= tf.get_variable("noise_bandwidth", dtype=tf.float32, shape=[embed_dim])
        decoder_input = noise

        decoder_input, decoder_self_attention_bias = transformer_prepare_decoder(decoder_input, hparams)

        return super(TransformerGAN, self).decode(decoder_input,
                                                  encoder_output,
                                                  encoder_decoder_attention_bias,
                                                  decoder_self_attention_bias,
                                                  hparams,
                                                  cache=cache)

    #                       nonpadding=nonpadding) #required when merged

    def model_fn_body(self, features):
        target_modality = self._problem_hparams.target_modality
        targets = features["targets"]
        inputs = features.get("inputs", None)
        hparams = self._hparams

        features["inputs"] = decay_gradient(features["inputs"], hparams.mle_decay_period)

        encoder_output, encoder_decoder_attention_bias = (None, None)
        if inputs is not None:
            target_space = features["target_space_id"]
            encoder_output, encoder_decoder_attention_bias = self.encode(inputs, target_space, hparams)

        targets = common_layers.flatten4d3d(targets)
        decoder_input, decoder_self_attention_bias = transformer_prepare_decoder(targets, hparams)

        decode_out = self.decode(decoder_input, encoder_output,
                                 encoder_decoder_attention_bias,
                                 decoder_self_attention_bias, hparams)

        with tf.variable_scope(tf.VariableScope(True)):
            with tf.variable_scope(target_modality.name + "/shared", reuse=True):
                embeddings = target_modality._get_weights()

        discrim_features = tf.gather(pinv_and_transpose(embeddings), features["targets_raw"])

        discrim_features = tf.reshape(discrim_features, tf.shape(features["targets"]))

        d_real = discriminator(discrim_features, features["inputs"], usage="real", hparams=self._hparams)
        d_fake = discriminator(decode_out, features["inputs"], usage="fake", hparams=self._hparams, reuse=True)

        tf.summary.scalar("real_score", tf.reduce_mean(d_real))
        tf.summary.scalar("gen_score", tf.reduce_mean(d_fake))

        d_loss = tf.reduce_mean(d_fake - d_real)

        alpha = tf.random_uniform(shape=[tf.shape(discrim_features)[0], 1, 1, 1], minval=0., maxval=1.)

        differences = tf.stop_gradient(decode_out - discrim_features)
        interpolates = tf.stop_gradient(discrim_features) + (alpha * differences)
        gradients = tf.gradients(
            discriminator(interpolates, features["inputs"], usage="gp", hparams=self._hparams, reuse=True),
            [interpolates])[0]

        slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), reduction_indices=[1, 2, 3]))
        gradient_penalty = tf.reduce_mean((slopes - 1.) ** 2)

        losses = {
            "discriminator": d_loss,
            "lipschitz-penalty": gradient_penalty * hparams.lipschitz_mult
        }

        return decode_out, losses

    def _fast_decode(self,
                     features,
                     decode_length,
                     beam_size=1,
                     top_beams=1,
                     alpha=1.0):
        """Fast decoding.

        Implements decoding,

        Args:
        features: a map of string to model  features.
        decode_length: an integer.  How many additional timesteps to decode.
        beam_size: number of beams.
        top_beams: an integer. How many of the beams to return.
        alpha: Float that controls the length penalty. larger the alpha, stronger
        the preference for slonger translations.

        Returns:
            samples: an integer `Tensor`. Top samples from the beam search

        Raises:
            NotImplementedError: If there are multiple data shards.
        """
        if top_beams * beam_size != 1:
            raise NotImplementedError("top beams and beam size must be 1")
        modality = self._problem_hparams.target_modality
        inputs = features["inputs"]

        batch_size = tf.shape(inputs)[0]
        decode_length = tf.shape(inputs)[1] + decode_length
        initial_ids = tf.zeros([batch_size, decode_length, 1, modality._body_input_depth], dtype=tf.float32)

        features["targets"] = initial_ids
        features["targets_raw"] = tf.zeros([batch_size, decode_length, 1, 1], dtype=tf.int32)

        with tf.variable_scope(modality.name, reuse=None):
            features["inputs"] = modality.bottom(features["inputs"])

        with tf.variable_scope("body", reuse=None):
            body_out = self.model_fn_body(features)

        with tf.variable_scope(modality.name, reuse=None):
            logits = modality.top(*body_out)

        return common_layers.sample_with_temperature(logits, 0.0), None


@registry.register_hparams
def transformer_gan_base():
    hparams = transformer_base_single_gpu()
    hparams.input_modalities = "inputs:symbol:GAN"
    hparams.target_modality = "symbol:GAN"
    hparams.batch_size = 1024
    hparams.learning_rate_decay_scheme = "none"
    hparams.optimizer = "Adam"
    hparams.summarize_grads = True
    hparams.clip_grad_norm = 1000.0
    hparams.add_hparam("num_compress_steps", 2)
    hparams.add_hparam("num_decode_steps", 0)
    hparams.add_hparam("discrim_grad_mul", 0.01)
    hparams.add_hparam("gan_label_smoothing", 1)
    hparams.add_hparam("step_interval", 1)
    hparams.add_hparam("warmup_steps", 4000)
    hparams.add_hparam("mle_decay_period", 500000)
    hparams.add_hparam("lipschitz_mult", 1e5)
    return hparams


def decay_gradient(outputs, decay_period):
    masking = common_layers.inverse_lin_decay(decay_period)
    masking = tf.minimum(tf.maximum(masking, 0.0), 1.0)
    tf.summary.scalar("loss_mask", masking)
    return tf.stop_gradient(masking * outputs) + (1.0 - masking) * outputs


@registry.register_symbol_modality("GAN")
class GANSymbolModality(modalities.SymbolModality):
    def loss(self, *args, **kwargs):
        loss, weights = super(GANSymbolModality, self).loss(*args, **kwargs)
        return decay_gradient(loss), decay_gradient(weights, self._model_hparams.mle_decay_period)
