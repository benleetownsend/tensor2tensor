from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Dependency imports

from tensor2tensor.layers import common_hparams, modalities
from tensor2tensor.layers import common_layers
from tensor2tensor.utils import registry
from tensor2tensor.models import transformer_vae
from tensor2tensor.models.transformer import Transformer, transformer_encoder, transformer_prepare_encoder, transformer_base_single_gpu
from tensor2tensor.utils import modality
import tensorflow as tf



def reverse_grad(x):
    return tf.stop_gradient(2 * x) - x

def discriminator(embedded_trans, embedded_context,target_space_id,  hparams, reuse=False):
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
    return tf.nn.l2_loss(truth_summed-predi_summed) / num_tokens
    
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
            context_embed = tf.sigmoid(model_fn_body(embedded_context)[:,1])
        with tf.variable_scope("response", reuse=reuse):
            trans_embed = tf.sigmoid(model_fn_body(embedded_trans)[:,1])
        discrim_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=scope.name)
    tf.logging.info("Discrim vars: "+str(discrim_vars))
#    clip = [v.assign(tf.clip_by_value(v, -1.0, 1.0)) for v in discrim_vars]

#    with tf.control_dependencies(clip):
    return tf.reduce_mean(tf.matmul(context_embed, trans_embed, transpose_b=True))


def stop_gradient_dict(d):
    new_dict = dict()
    for key, val in d.items():
        new_dict[key] = tf.stop_gradient(val)
    return new_dict

@registry.register_model
class TransformerGAN(Transformer):
    def model_fn_body(self, features):
        #features = stop_gradient_dict(features)
        """TransformerGAN main model_fn."""
        outputs = super(TransformerGAN, self).model_fn_body(features)
        # train = self._hparams.mode == tf.estimator.ModeKeys.TRAIN

        g_sample = outputs

        self.hparams.epsilon = 1e-5

        discrim_features = features["targets"]
        discrim_features = tf.stop_gradient(discrim_features) + tf.random_normal(tf.shape(discrim_features), mean=0.0, stddev=0.0001)
        d_real = discriminator(discrim_features, tf.stop_gradient(features["inputs"]), features["target_space_id"],
                               self._hparams)
        d_fake = discriminator(reverse_grad(g_sample), tf.stop_gradient(features["inputs"]), features["target_space_id"],
                               self._hparams, reuse=True)

#        d_real = tf.Print(d_real, [d_real, d_fake])

        tf.summary.scalar("real_score", tf.reduce_mean(d_real))
        tf.summary.scalar("gen_score", tf.reduce_mean(d_fake))

        d_loss = tf.reduce_mean(d_fake - d_real)
        g_loss = tf.stop_gradient(tf.reduce_mean(-d_fake))
        c_loss = cbow_loss(features["targets"], g_sample)

        # WGAN lipschitz-penalty
        alpha = tf.random_uniform(
                shape=[tf.shape(discrim_features)[0],1,1,1],
                minval=0.,
                maxval=1.
        )
        differences = tf.stop_gradient(g_sample - discrim_features)
        interpolates = discrim_features + (alpha*differences)
        gradients = tf.gradients(discriminator(interpolates, tf.stop_gradient(features["inputs"]), features["target_space_id"],
                                               self._hparams, reuse=True), [interpolates])[0]
        slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), reduction_indices=[1,2,3]))
        gradient_penalty = tf.reduce_mean((slopes-1.)**2)

        losses = dict()
        losses["discriminator"] = d_loss
        losses["generator"] = g_loss
        losses["lipschitz-penalty"] = gradient_penalty * 100
        losses["cbow_loss"] = c_loss
        
        return  outputs, losses

@registry.register_hparams
def transformer_gan_base():
    hparams = transformer_base_single_gpu()
    hparams.input_modalities="inputs:symbol:GAN"
    hparams.target_modality="symbol:GAN"
    hparams.batch_size = 1024
    hparams.learning_rate_decay_scheme = "none"
    hparams.optimizer="SGD"
    hparams.summarize_grads = True
    hparams.clip_grad_norm = 10.0
    hparams.add_hparam("num_compress_steps", 2)
    return hparams


def decay_gradient(outputs):
    masking = common_layers.inverse_lin_decay(200000)
    masking *= common_layers.inverse_exp_decay(50000)  # Not much at start.
    masking = tf.minimum(tf.maximum(masking, 0.0), 1.0)
    tf.summary.scalar("loss_mask", masking)
    return tf.stop_gradient(masking * outputs) + (1.0 - masking) * outputs 

@registry.register_symbol_modality("GAN")
class GANSymbolModality(modalities.SymbolModality):
    def loss(self, *args, **kwargs):
#        return tf.constant(0.0), tf.constant(0.0)
        loss, weights =  super(GANSymbolModality, self).loss(*args, **kwargs)
        return decay_gradient(loss), decay_gradient(weights)
        
    
