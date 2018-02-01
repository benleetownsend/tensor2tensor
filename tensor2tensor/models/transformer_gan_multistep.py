from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Dependency imports

from tensor2tensor.layers import common_hparams, modalities
from tensor2tensor.layers import common_layers
from tensor2tensor.utils import registry
from tensor2tensor.models import transformer_vae
from tensor2tensor.models.transformer import Transformer, transformer_encoder, transformer_prepare_encoder, \
    transformer_base_single_gpu, transformer_prepare_decoder
from tensor2tensor.utils import modality
import tensorflow as tf
import numpy as np

def reverse_grad(x):
    return tf.stop_gradient(2 * x) - x

def reverse_and_reduce_gradient(x, grad_mul=0.01, step_interval=None, warmup=None):
    warmup = warmup or 0

    if step_interval is not None:
        global_step = tf.train.get_or_create_global_step()
        gating_multiplier = tf.to_float(
            tf.logical_and(tf.equal(tf.mod(global_step, step_interval), 0), tf.greater(global_step, warmup)))
        grad_mul *= gating_multiplier

    return tf.stop_gradient(x + grad_mul * x) - grad_mul * x


def soft_embed(x, embedding, batch_size, embed_size, vocab_size):
    """Softmax x and embed."""
    x = tf.reshape(x, [-1, vocab_size])
    x = tf.matmul(x, embedding)
    return tf.reshape(x, [batch_size, -1, 1, embed_size])


def discriminator(embedded_trans, embedded_context, target_space_id, hparams, reuse=False):
    with tf.variable_scope("discriminator", reuse=reuse):
        h0, i0 = common_layers.pad_to_same_length(
            embedded_trans, embedded_context, final_length_divisible_by=16)
        h0 = tf.concat([h0, i0], axis=-1)
        h0 = tf.layers.dense(h0, hparams.hidden_size, name="io")
        h1 = transformer_vae.compress(h0, None, False, hparams, "compress1")
        h2 = transformer_vae.compress(h1, None, False, hparams, "compress2")
        res_dense = tf.reduce_mean(h2, axis=[1, 2])
        res_single = tf.squeeze(tf.layers.dense(res_dense, 1), axis=-1)
        return res_single


def cbow_loss(truth, predicted):
    truth = common_layers.flatten4d3d(truth)
    predicted = common_layers.flatten4d3d(predicted)
    truth_summed = tf.cumsum(truth, reverse=True, axis=-2)
    predi_summed = tf.cumsum(predicted, reverse=True, axis=-2)
    num_tokens = tf.to_float(tf.reduce_prod(tf.shape(truth)))
    return tf.nn.l2_loss(truth_summed - predi_summed) / num_tokens


def discriminator_(embedded_trans, embedded_context, target_space_id, hparams, reuse=False):
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

    with tf.variable_scope("discriminator", reuse=reuse) as scope:
        with tf.variable_scope("context", reuse=reuse):
            context_embed = tf.sigmoid(model_fn_body(embedded_context)[:, 1])
        with tf.variable_scope("response", reuse=reuse):
            trans_embed = tf.sigmoid(model_fn_body(embedded_trans)[:, 1])
        discrim_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=scope.name)
    tf.logging.info("Discrim vars: " + str(discrim_vars))
    #    clip = [v.assign(tf.clip_by_value(v, -1.0, 1.0)) for v in discrim_vars]

    #    with tf.control_dependencies(clip):
    return tf.reduce_mean(tf.matmul(context_embed, trans_embed, transpose_b=True))

#def setup_lazy_pinv(threshold=0.1):
#    state = None
#    cache = None


class Lazy_pinv:
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
    return tf.stop_gradient(tf.transpose(tf.py_func(Lazy_pinv(), [inp], tf.float32)))

def stop_gradient_dict(d):
    new_dict = dict()
    for key, val in d.items():
        new_dict[key] = tf.stop_gradient(val)
    return new_dict


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
        noise = tf.random_uniform(
            shape=tf.shape(decoder_input),
            minval=-0.5,
            maxval=0.5
        )
        noise *= tf.get_variable("noise_bandwidth", dtype=tf.float32, shape=[embed_dim])
        decoder_input = noise

        decoder_input, decoder_self_attention_bias = transformer_prepare_decoder(
                        decoder_input, hparams)

        return super().decode(decoder_input,
                              encoder_output,
                              encoder_decoder_attention_bias,
                              decoder_self_attention_bias,
                              hparams,
                              cache=cache)

    #                       nonpadding=nonpadding) #required when merged

    def model_fn_body(self, features):
        import traceback

        try:
            raise TypeError("Oups!")
        except Exception:
            traceback.print_exc()
                            
        features["inputs"] = decay_gradient(features["inputs"])
        discrim_features = features["targets"]
        hparams = self._hparams

        is_training = self._hparams.mode == tf.estimator.ModeKeys.TRAIN
        # START FROM ATTN
        inputs = features.get("inputs")
        encoder_output, encoder_decoder_attention_bias = (None, None)
        if inputs is not None:
            target_space = features["target_space_id"]
            encoder_output, encoder_decoder_attention_bias = self.encode(
                inputs, target_space, hparams)

        targets = features["targets"]

        targets = common_layers.flatten4d3d(targets)
        decoder_input, decoder_self_attention_bias = transformer_prepare_decoder(
            targets, hparams)

        outputs_for_MLE = self.decode(decoder_input, encoder_output,
                                      encoder_decoder_attention_bias,
                                      decoder_self_attention_bias, hparams)

        targets = outputs_for_MLE
        if not is_training or hparams.num_decode_steps == 0:
            hparams.num_decode_steps = 0
            outputs = outputs_for_MLE
        else:
            outputs = None
        for _ in range(hparams.num_decode_steps):
            targets = common_layers.flatten4d3d(targets)
            decoder_input, decoder_self_attention_bias = transformer_prepare_decoder(
                targets, hparams)

            outputs = self.decode(decoder_input, encoder_output,
                                  encoder_decoder_attention_bias,
                                  decoder_self_attention_bias, hparams)
            targets = outputs
            # TODO (BEN) stop grads???

            # TODO (BEN) de-embed outputs and set as targets. (maybe this will suffice)

        # END FROM ATTN
        self.hparams.epsilon = 1e-5

        gsample_embedded = outputs

        
        target_modality = self._problem_hparams.target_modality
        with tf.variable_scope(tf.VariableScope(True)):
            with tf.variable_scope(target_modality.name+"/shared", reuse=True):
                embeddings = target_modality._get_weights()
        
        discrim_features = tf.gather(pinv_and_transpose(embeddings), features["targets_raw"])

        discrim_features = tf.reshape(discrim_features, tf.shape(features["targets"]))

#        discrim_features.set_shape(features["targets"].shape)
#        discrim_features = tf.squeeze(discrim_features, -2)
        d_real = discriminator(tf.stop_gradient(discrim_features), tf.stop_gradient(features["inputs"]),
                               features["target_space_id"], self._hparams)
        d_fake = discriminator(
            reverse_and_reduce_gradient(gsample_embedded, self._hparams.discrim_grad_mul, self.hparams.step_interval,
                                        self.hparams.warmup_steps), tf.stop_gradient(features["inputs"]),
            features["target_space_id"], self._hparams, reuse=True)

        tf.summary.scalar("real_score", tf.reduce_mean(d_real))
        tf.summary.scalar("gen_score", tf.reduce_mean(d_fake))

        d_loss = tf.reduce_mean(d_fake - d_real)

        alpha = tf.random_uniform(
            shape=[tf.shape(discrim_features)[0], 1, 1, 1],
            minval=0.,
            maxval=1.
        )

        differences = tf.stop_gradient(gsample_embedded - discrim_features)
        interpolates = tf.stop_gradient(discrim_features) + (alpha * differences)
        gradients = tf.gradients(discriminator(
            interpolates, tf.stop_gradient(features["inputs"]),
            features["target_space_id"], self._hparams, reuse=True), [interpolates])[0]

        slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), reduction_indices=[1, 2, 3]))
        gradient_penalty = tf.reduce_mean((slopes - 1.) ** 2)

        losses = dict()
        losses["discriminator"] = d_loss
        losses["lipschitz-penalty"] = gradient_penalty * 1e5

        return outputs_for_MLE, losses

    def _fast_decode(self,
                     features,
                     decode_length,
                     beam_size=1,
                     top_beams=1,
                     alpha=1.0):
          """Fast decoding.

          Implements both greedy and beam search decoding, uses beam search iff
          beam_size > 1, otherwise beam search related arguments are ignored.
          
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
          target_modality = self._problem_hparams.target_modality
          inputs = features["inputs"]
          with tf.variable_scope(target_modality.name, reuse=None):
              features["inputs"] = target_modality.bottom(features["inputs"])
          batch_size = tf.shape(inputs)[0]
          decode_length = tf.shape(inputs)[1] + decode_length
          initial_ids = tf.zeros([batch_size, decode_length, 1, target_modality._body_input_depth], dtype=tf.float32)

          features["targets"] = initial_ids
          features["targets_raw"] = tf.zeros([batch_size, decode_length, 1, 1], dtype=tf.int32)

#          with tf.variable_scope(target_modality.name+"/shared", reuse=None):
#                              embeddings = target_modality._get_weights()                  

          with tf.variable_scope("body", reuse=None):
              body_out = self.model_fn_body(features)
              

          with tf.variable_scope(target_modality.name, reuse=None):
              logits = target_modality.top(*body_out)

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
    return hparams


def decay_gradient(outputs):
    masking = common_layers.inverse_lin_decay(500000)
    masking *= common_layers.inverse_exp_decay(10000)  # Not much at start.
    masking = tf.minimum(tf.maximum(masking, 0.0), 1.0)
    tf.summary.scalar("loss_mask", masking)
    return tf.stop_gradient(masking * outputs) + (1.0 - masking) * outputs


@registry.register_symbol_modality("GAN")
class GANSymbolModality(modalities.SymbolModality):
    def loss(self, *args, **kwargs):
        loss, weights = super(GANSymbolModality, self).loss(*args, **kwargs)
        return decay_gradient(loss), decay_gradient(weights)
