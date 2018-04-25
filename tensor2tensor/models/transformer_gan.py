from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Dependency imports
from six.moves import xrange  # pylint: disable=redefined-builtin

from tensor2tensor.layers import common_layers
from tensor2tensor.utils import expert_utils as eu
from tensor2tensor.layers import modalities
from tensor2tensor.layers import common_layers
from tensor2tensor.layers import common_attention
from tensor2tensor.utils import registry
from tensor2tensor.models import transformer_vae
from tensor2tensor.models.transformer import Transformer, transformer_base_single_gpu, transformer_prepare_decoder
from tensorflow.python.ops.rnn import _transpose_batch_time
# from tensor2tensor.fertility_model.alignments_to_fertility import FertilityModel
from tensor2tensor.models.fertility import FertilityModel as DQNFertility
from tensor2tensor.models.fertility import tf_decomposed_gleu
from tensor2tensor.models.language.output_smooth import SmoothOutput
import tensorflow as tf
import numpy as np
import pickle


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


def highway(input_, size, num_layers=1, bias=-2.0, f=tf.nn.relu, scope='Highway'):
    """Highway Network (cf. http://arxiv.org/abs/1505.00387).                                                                                                                                                                                                                                    
    t = sigmoid(Wy + b)                                                                                                                                                                                                                                                                          
    z = t * g(Wy + b) + (1 - t) * y                                                                                                                                                                                                                                                              
    where g is nonlinearity, t is transform gate, and (1 - t) is carry gate.                                                                                                                                                                                                                     
    """

    with tf.variable_scope(scope):
        for idx in range(num_layers):
            g = f(linear(input_, size, scope='highway_lin_%d' % idx))
            t = f(linear(g, size, scope='highway_gate_%d' % idx))
            output = g
        return output


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


def discriminator(embedded_trans, embedded_context, hparams, usage, reuse=False):
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
            h_highway = highway(h_pool_flat, h_pool_flat.get_shape()[1], 1, 0, f=leaky_relu)

        with tf.variable_scope("dropout", reuse=reuse):
            h_drop = tf.nn.dropout(h_highway, 1.0 - hparams.discrim_dropout)

        with tf.variable_scope("output", reuse=reuse):
            W = tf.get_variable("W", shape=[num_filters_total, 1])
            b = tf.get_variable("b", shape=[1])
            score = tf.nn.xw_plus_b(h_drop, W, b, name="scores")
        return score


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

        return super(TransformerGAN, self).decode(decoder_input,
                                                  encoder_output,
                                                  encoder_decoder_attention_bias,
                                                  decoder_self_attention_bias,
                                                  hparams,
                                                  cache=cache)

    def model_fn_body(self, features):
        target_modality = self._problem_hparams.target_modality
        fert_filename = self._hparams.fertility_filename

        original_targets = features["targets"]
        inputs = features.get("inputs", None)
        hparams = self._hparams

        encoder_output, encoder_decoder_attention_bias = (None, None)
        if inputs is not None:
            target_space = features["target_space_id"]

            encoder_output, encoder_decoder_attention_bias = self.encode(inputs, target_space, hparams)

        if fert_filename is not None:
            tf.logging.info("Loading Fertility Model")
            fert_model = FertilityModel(fert_filename,
                                        self._hparams.reinforce_delta if self._hparams.mode == tf.estimator.ModeKeys.TRAIN else 0.0)
            features["inputs_fert_raw"], fertilities = tf.py_func(fert_model.fertilize, [features["inputs_raw"]],
                                                                  [tf.int32, tf.int32])
            features["inputs_fert_raw"] = tf.reshape(features["inputs_fert_raw"], tf.shape(features["inputs_raw"]))

            def reinforce(reward):
                return tf.py_func(fert_model.reinforce, [features["inputs_raw"], fertilities, reward], tf.int64)

        else:
            tf.logging.info("Using DQN fertility model")
            fert_model = DQNFertility(max_fertility=5, hparams=self._hparams)
            features["inputs_fert_raw"], fertilities = fert_model.fertilise(inputs, features["inputs_raw"])
            features["inputs_fert_raw"] = tf.reshape(features["inputs_fert_raw"], tf.shape(features["inputs_raw"]))

            def reinforce(reward):
                return fert_model.reinforce_op(fertilities, reward)

        with tf.variable_scope(tf.VariableScope(True)):
            with tf.variable_scope(target_modality.name, reuse=True):
                features["inputs_fert"] = target_modality.bottom(features["inputs_fert_raw"])

        inputs = common_layers.flatten4d3d(features["inputs_fert"])

        decoder_self_attention_bias = (
            common_attention.attention_bias_lower_triangle(tf.shape(inputs)[1]))
        decoder_input = common_attention.add_timing_signal_1d(tf.nn.dropout(inputs, 1 - hparams.z_temp))

        decode_out = self.decode(decoder_input, encoder_output,
                                 encoder_decoder_attention_bias,
                                 tf.ones_like(decoder_self_attention_bias), hparams)

        with tf.variable_scope(tf.VariableScope(True)):
            with tf.variable_scope(target_modality.name + "/shared", reuse=True):
                embeddings = target_modality._get_weights()

        trans_embeddings = tf.get_variable("trans_embed", shape=embeddings.get_shape().as_list())

        discrim_features = tf.gather(trans_embeddings, features["targets_raw"])
        discrim_features = tf.reshape(discrim_features, tf.shape(original_targets))

        projected_truth = tf.matmul(tf.reshape(discrim_features, [-1, hparams.hidden_size]),
                                    tf.stop_gradient(embeddings), transpose_b=True)

        trans_embed_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=tf.reshape(features["targets_raw"], [-1]),
            logits=projected_truth,
            name="trans_embed_loss"
        )
        discrim_features += tf.random_normal(tf.shape(discrim_features), stddev=0.1)

        d_real = discriminator(discrim_features, features["inputs"], usage="real", hparams=self._hparams)
        tf.summary.scalar("real_score", tf.reduce_mean(d_real))

        gradient_penalty = 0.0
        if hparams.ganmode == "wgan-gp":
            alpha = tf.random_uniform(shape=[tf.shape(discrim_features)[0], 1, 1, 1], minval=0., maxval=1.)

            differences = tf.stop_gradient(decode_out - discrim_features)
            interpolates = tf.stop_gradient(discrim_features) + (alpha * differences)
            gradients = tf.gradients(
                discriminator(interpolates, features["inputs"], usage="gp", hparams=self._hparams, reuse=True),
                [interpolates])[0]

            slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), reduction_indices=[1, 2, 3]))
            gradient_penalty = tf.reduce_mean((slopes - 1.) ** 2) * hparams.lipschitz_mult

        d_fake = discriminator(decode_out, features["inputs"], usage="fake", hparams=self._hparams, reuse=True)
        tf.summary.scalar("gen_score", tf.reduce_mean(d_fake))
        d_loss = tf.reduce_mean(d_fake - d_real)
        d_real_cycled = discriminator(tf.random_shuffle(discrim_features), features["inputs"], usage="real",
                                      hparams=self._hparams, reuse=True)
        d_loss_cycle = tf.reduce_mean(d_real_cycled - d_real)

        if hparams.ganmode == "wgan":
            d_vars = [var for var in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES) if "discriminator" in var.name]
            clip_ops = []
            for var in d_vars:
                clip_bounds = [-.002, .002]
                clip_ops.append(
                    tf.assign(
                        var,
                        tf.clip_by_value(var, clip_bounds[0], clip_bounds[1])
                    )
                )
            with tf.control_dependencies(clip_ops):
                d_loss = tf.identity(d_loss)

        with tf.variable_scope(tf.VariableScope(True)):
            with tf.variable_scope(target_modality.name, reuse=True):
                logits = target_modality.top(decode_out, None)
                sample = common_layers.sample_with_temperature(logits, 0.0)

        losses = {
            "discriminator": d_loss,
            "lipschitz-penalty": gradient_penalty,
            "trans_embed_loss": trans_embed_loss,
            "semantic_reg": d_loss_cycle * 150,
            "reinforce_fert": tf.reduce_mean(reinforce(d_fake * hparams.reinforce_delta))
        }
        if self._hparams.mode == tf.estimator.ModeKeys.PREDICT:
            return decode_out, d_fake

        return decode_out, losses

    def model_fn(self, features, **kwargs):
        features["inputs"], features["targets"] = common_layers.pad_to_same_length(features["inputs"],
                                                                                   features["targets"])
        return super(TransformerGAN, self).model_fn(features, **kwargs)

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
        raw_inputs = features["inputs"]
        batch_sz = tf.shape(raw_inputs)[0]

        modality = self._problem_hparams.target_modality

        features["inputs"] = tf.tile(features["inputs"],
                                     [beam_size] + [1] * (len(features["inputs"].get_shape().as_list()) - 1))
        features["inputs_raw"] = inputs = features["inputs"]
        inputs = tf.expand_dims(inputs, axis=1)
        if len(inputs.shape) < 5:
            inputs = tf.expand_dims(inputs, axis=4)
        s = tf.shape(inputs)
        inputs = tf.reshape(inputs, [s[0] * s[1], s[2], s[3], s[4]])
        features["inputs_raw"] = inputs

        batch_size = tf.shape(inputs)[0]
        decode_length = tf.shape(inputs)[1] + decode_length
        initial_ids = tf.zeros([batch_size, decode_length, 1, modality._body_input_depth], dtype=tf.float32)

        features["targets"] = initial_ids
        features["targets_raw"] = tf.zeros([batch_size, decode_length, 1, 1], dtype=tf.int32)

        features["targets_raw"], inputs = common_layers.pad_to_same_length(features["targets_raw"], inputs)

        smoother_decoder = SmoothOutput(self._hparams.lang_model_file, self._hparams.lang_model_data,
                                        self._hparams.problems[0].vocabulary["targets"])

        with tf.variable_scope(modality.name, reuse=None):
            features["inputs"] = modality.bottom(inputs)

        with tf.variable_scope("body", reuse=None):
            feats, losses = self.model_fn_body(features)
            feats = tf.reshape(feats, tf.concat([[beam_size, batch_sz], tf.shape(feats)[1:]], axis=0))
            losses = tf.reshape(losses, tf.concat([[beam_size, batch_sz], tf.shape(losses)[1:]], axis=0))

        top_for_each_batch = tf.squeeze(tf.argmax(losses, 0, output_type=tf.int32), [-1])

        indices = tf.transpose(tf.stack([top_for_each_batch, tf.range(batch_sz)]))
        gathered = tf.gather_nd(feats, indices)
        body_out = tf.reshape(gathered, tf.shape(feats)[1:]), None

        body_out = tf.Print(body_out[0], [body_out[0]]), None
        with tf.variable_scope(modality.name, reuse=None):
            logits = modality.top(*body_out)

        features["inputs"] = raw_inputs

        return smoother_decoder.decode_sequence(logits), None


@registry.register_hparams
def transformer_gan_base():
    hparams = transformer_base_single_gpu()
    hparams.input_modalities = "inputs:symbol:GAN"
    hparams.target_modality = "symbol:GAN"
    hparams.batch_size = 1024
    hparams.learning_rate = 1e-5
    hparams.learning_rate_decay_scheme = "none"
    hparams.optimizer = "RMSProp"
    hparams.summarize_grads = False
    hparams.clip_grad_norm = 1000.0
    hparams.num_decoder_layers = 6
    hparams.num_encoder_layers = 6
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
    hparams.add_hparam("warmup_steps", 2150001)
    hparams.add_hparam("mle_decay_period", 2150000)
    hparams.add_hparam("embed_decay_period", 2150000)
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
def transformer_gan_base_ro():
    hparams = transformer_gan_base()
    hparams.embedding_file = None
    hparams.z_temp = 0
    hparams.fertility_filename = "translate_roen_wmt8k.alignfertility_model.pkl"
    hparams.add_hparam("lang_model_file", "/root/code/t2t_data/lang_model_small.eng.pkl")
    hparams.add_hparam("lang_model_data", "/root/code/t2t_data/t2t_datagen/ro-en-lang_data.txt")
    return hparams


@registry.register_hparams
def transformer_gan_base_ro_reinforce():
    hparams = transformer_gan_base_ro()
    hparams.embedding_file = None
    hparams.z_temp = 0
    hparams.fertility_filename = None
    return hparams


@registry.register_hparams
def transformer_gan_fat_ro_reinforce():
    hparams = transformer_gan_base_ro_reinforce()
    hparams.num_heads = 16
    hparams.hidden_size = 1024
    hparams.filter_size = 4096

    return hparams


@registry.register_hparams
def transformer_gan_german():
    hparams = transformer_gan_base()
    hparams.embedding_file = None
    hparams.z_temp = 0
    hparams.fertility_filename = None
    hparams.learning_rate = 1e-6
    hparams.reinforce_delta = 1e-1
    hparams.add_hparam("lang_model_file", "/root/code/t2t_data/lang_model_small.germ.pkl")
    hparams.add_hparam("lang_model_data", "/root/code/t2t_data/t2t_datagen/german_lang_model_data.txt")
    hparams.ganmode = "wgan-gp"
    return hparams


@registry.register_hparams
def transformer_gan_base_mini():
    hparams = transformer_gan_base()
    hparams.hidden_size = 128
    return hparams


def decay_gradient(outputs, decay_period, final_val=1.00, summarize=True):
    masking = common_layers.inverse_lin_decay(decay_period)
    masking = tf.minimum(tf.maximum(masking, 0.0), final_val)
    if summarize:
        tf.summary.scalar("loss_mask", masking)
    return tf.stop_gradient(masking * outputs) + (1.0 - masking) * outputs


@registry.register_symbol_modality("GAN")
class GANSymbolModality(modalities.SymbolModality):
    def _get_weights(self, hidden_dim=None):
        if self._model_hparams.embedding_file is None:
            initialiser = lambda name: tf.random_normal_initializer(0.0, hidden_dim ** -0.5)
        else:
            with open(self._model_hparams.embedding_file, "rb") as fp:
                embeddings = pickle.load(fp)
            tf.logging.info("Loading embeddings from file")
            initialiser = lambda name: tf.constant(embeddings["symbol_modality_27927_512/shared/" + name + ":0"])

        if hidden_dim is None:
            hidden_dim = self._body_input_depth
        num_shards = self._model_hparams.symbol_modality_num_shards
        shards = []
        for i in xrange(num_shards):
            shard_size = (self._vocab_size // num_shards) + (
                1 if i < self._vocab_size % num_shards else 0)
            var_name = "weights_%d" % i
            shards.append(
                tf.get_variable(
                    var_name,
                    initializer=initialiser(var_name),
                    shape=None if self._model_hparams.embedding_file is not None else [shard_size, hidden_dim]))
        if num_shards == 1:
            ret = shards[0]
        else:
            ret = tf.concat(shards, 0)
        ret = eu.convert_gradient_to_tensor(ret)

        if type(ret) == list:
            weights = [decay_gradient(w, self._model_hparams.embed_decay_period, summarize=False) for w in ret]
        else:
            weights = decay_gradient(ret, self._model_hparams.embed_decay_period, summarize=False)
        return weights

    #    @property
    #    def targets_weights_fn(self):
    #        return common_layers.weights_all

    def loss(self, *args, **kwargs):
        loss, weights = super(GANSymbolModality, self).loss(*args, **kwargs)
        return decay_gradient(loss, self._model_hparams.mle_decay_period, summarize=False), weights
