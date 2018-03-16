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
from tensor2tensor.fertility_model.alignments_to_fertility import FertilityModel

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
            t = tf.sigmoid(linear(input_, size, scope='highway_gate_%d' % idx) + bias)
            output = t * g + (1. - t) * input_
            input_ = output
        return output

def lstm_bid_encoder(inputs, hparams, train, name):
    """Bidirectional LSTM for encoding inputs that are [batch x time x size]."""

    def dropout_lstm_cell():
        return tf.contrib.rnn.DropoutWrapper(
            tf.contrib.rnn.BasicLSTMCell(hparams.hidden_size//2),
            input_keep_prob=1.0 - hparams.dropout * tf.to_float(train))

    with tf.variable_scope(name):
        cell_fw = tf.contrib.rnn.MultiRNNCell(
            [dropout_lstm_cell() for _ in range(hparams.fertility_cells)])

        cell_bw = tf.contrib.rnn.MultiRNNCell(
            [dropout_lstm_cell() for _ in range(hparams.fertility_cells)])

        ((encoder_fw_outputs, encoder_bw_outputs),
         (encoder_fw_state, encoder_bw_state)) = tf.nn.bidirectional_dynamic_rnn(
            cell_fw=cell_fw,
            cell_bw=cell_bw,
            inputs=inputs,
             swap_memory=False,
            dtype=tf.float32,
            time_major=False)

        encoder_outputs = tf.concat((encoder_fw_outputs, encoder_bw_outputs), 2)
        encoder_states = []

        for i in range(hparams.fertility_cells):
            if isinstance(encoder_fw_state[i], tf.contrib.rnn.LSTMStateTuple):
                encoder_state_c = tf.concat(
                    values=(encoder_fw_state[i].c, encoder_bw_state[i].c),
                    axis=1,
                    name="encoder_fw_state_c")
                encoder_state_h = tf.concat(
                    values=(encoder_fw_state[i].h, encoder_bw_state[i].h),
                    axis=1,
                    name="encoder_fw_state_h")
                encoder_state = tf.contrib.rnn.LSTMStateTuple(
                    c=encoder_state_c, h=encoder_state_h)
            elif isinstance(encoder_fw_state[i], tf.Tensor):
                encoder_state = tf.concat(
                    values=(encoder_fw_state[i], encoder_bw_state[i]),
                    axis=1,
                    name="bidirectional_concat")

            encoder_states.append(encoder_state)

        encoder_states = tuple(encoder_states)
        return encoder_outputs, encoder_states


def reverse_grad(x):
    return tf.stop_gradient(2 * x) - x


def fertility_model(inputs, hparams, modality, train, name):
    """Run LSTM cell on inputs, assuming they are [batch x time x size]."""
    inputs = inputs_for_attn = tf.squeeze(inputs, 2)
    inputs = tf.reverse(inputs, [1])

    def get_decoder_loop_fn(sequence_length, initial_state):
        def loop_fn(time, cell_output, cell_state, loop_state):

            emit_output = cell_output
            if cell_output is None:
                next_cell_state = initial_state
                next_input = tf.zeros(shape=[tf.shape(inputs)[0], 512], dtype=tf.float32)#tf.random_uniform(shape=[tf.shape(inputs)[0], 512], minval=-0.00, maxval=0.00,
#                                               dtype=tf.float32)  # GO
            else:
#                with tf.variable_scope(tf.VariableScope(True)):
#                    with tf.variable_scope(modality.name, reuse=True):
                next_input = cell_output    
#                        word_probs = tf.nn.softmax(modality.top(cell_output, None), dim=-1)
#                        word_ids = common_layers.sample_with_temperature(word_probs, hparams.z_temp)
#                        word_ids = tensor_reshape = tf.reshape(word_ids, [-1, 1, 1, 1])
#                        next_input = tf.squeeze(modality.bottom(word_ids), [1, 2])
                next_cell_state = cell_state

            elements_finished = (time >= sequence_length)
            next_loop_state = None
            return (elements_finished, next_input, next_cell_state, emit_output, next_loop_state)

        return loop_fn

    def dropout_lstm_cell():
        return tf.contrib.rnn.DropoutWrapper(
            tf.contrib.rnn.BasicLSTMCell(hparams.hidden_size),
            input_keep_prob=1.0 - hparams.dropout * tf.to_float(train))

    with tf.variable_scope("decoder_lstms"):
        decoder_layers = [dropout_lstm_cell() for _ in range(hparams.fertility_cells)]

    attention_mechanism_class = tf.contrib.seq2seq.LuongAttention

    with tf.variable_scope(name):
        encoder_outputs, encoder_final_state = lstm_bid_encoder(inputs, hparams, train, name+"Encoder")

        attention_mechanism = attention_mechanism_class(hparams.hidden_size, inputs_for_attn)

        attn_cell = tf.contrib.seq2seq.AttentionWrapper(
            tf.nn.rnn_cell.MultiRNNCell(decoder_layers),
            attention_mechanism,
            attention_layer_size=hparams.hidden_size,
            output_attention=True)

        batch_size = inputs.get_shape()[0].value
        if batch_size is None:
            batch_size = tf.shape(inputs)[0]
        initial_state = attn_cell.zero_state(batch_size, tf.float32).clone(
                  cell_state=encoder_final_state)
        
        with tf.variable_scope("decoder_lstms"):
            outputs, _, _ = tf.nn.raw_rnn(attn_cell,
                                          get_decoder_loop_fn(tf.shape(inputs)[1], initial_state),
                                          swap_memory=True)
        outputs = _transpose_batch_time(outputs.stack())

        return tf.expand_dims(outputs, 2)


def reverse_and_reduce_gradient(x, grad_pen, hparams=None):
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
        grad_mul *= 1 / (1 + grad_pen)

        tf.summary.scalar("gen_grad_mul", grad_mul)

    return tf.stop_gradient(x + grad_mul * x) - grad_mul * x


def discriminator(embedded_trans, embedded_context, hparams, usage, reuse=False, grad_pen=None):
    """
    Usage in ["real", "fake", "gp"]
    """
    if embedded_context is not None:
        embedded_context = tf.stop_gradient(embedded_context)

    if usage == "real":
        embedded_trans = tf.stop_gradient(embedded_trans)
    elif usage == "fake":
        embedded_trans = reverse_and_reduce_gradient(embedded_trans, grad_pen, hparams)
    elif usage == "gp":
        embedded_trans = embedded_trans
    else:
        raise KeyError("usage not in real, fake or gp")

    filter_sizes = [1, 2, 3, 4, 6, 8, 10]
    num_filters = [500, 500, 500, 500, 200, 200, 100]
    h0, i0 = common_layers.pad_to_same_length(embedded_trans, embedded_context, final_length_divisible_by=max(filter_sizes))
        
    with tf.variable_scope("discriminator", reuse=reuse):
        h0 = tf.expand_dims(tf.squeeze(h0, -2), -1)
        i0 = tf.expand_dims(tf.squeeze(i0, -2), -1)
        
        sequence_length = h0.get_shape()[1]
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
                    h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
#                    pooled = tf.nn.max_pool(
#                        h,
#                        ksize=[1, hparams.max_length  -  filter_size + 1, 1, 1],
#                        strides=[1, 1, 1, 1],
#                        padding='VALID',
#                        name="pool")
                    pooled= tf.reduce_max(h, axis=1, keep_dims=True) 
                    pooled_outputs.append(pooled)
        num_filters_total = sum(num_filters) * 2
        h_pool = tf.concat(pooled_outputs, 3)
        h_pool_flat = tf.reshape(h_pool, [-1, num_filters_total])


        with tf.variable_scope("highway", reuse=reuse):
            h_highway = highway(h_pool_flat, h_pool_flat.get_shape()[1], 1, 0)
            
        with tf.variable_scope("dropout", reuse=reuse):
            h_drop = tf.nn.dropout(h_highway, 1.0 - hparams.discrim_dropout)
                
        with tf.variable_scope("output", reuse=reuse):
            W = tf.get_variable("W", shape=[num_filters_total, 1])
            b = tf.get_variable("b", shape=[1])
            score = tf.nn.xw_plus_b(h_drop, W, b, name="scores")
        return score


class LazyPinv:
    def __init__(self):
        self.cache = None
        self.state = None
        self.call_count = 0

    def __call__(self, inp):
        self.call_count += 1
        if self.call_count % 10000 == 0 or self.state is None or np.linalg.norm(self.state - inp) > 10.0:
            self.cache = np.linalg.pinv(inp) * 100 
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

#        embed_dim = self._hparams.hidden_size
#        noise = tf.random_uniform(shape=tf.shape(encoder_output), minval=-0.05, maxval=0.05)
#        noise *= tf.get_variable("noise_bandwidth", dtype=tf.float32, shape=[embed_dim])
#        encoder_output += noise
#        decoder_input = tf.nn.dropout(decoder_input, 0.75)
        return super(TransformerGAN, self).decode(decoder_input,
                                                  encoder_output,
                                                  encoder_decoder_attention_bias,
                                                  decoder_self_attention_bias,
                                                  hparams,
                                                  cache=cache)

    def model_fn_body(self, features):
        target_modality = self._problem_hparams.target_modality
        fert_filename = self._hparams.fertility_filename
        if fert_filename is not None:
            tf.logging.info("Loading Fertility Model")
            fert_model = FertilityModel(fert_filename)
            features["inputs_fert_raw"] = tf.py_func(fert_model.fertilize, [features["inputs_raw"]], tf.int32)
            features["inputs_fert_raw"] =  tf.reshape(features["inputs_fert_raw"], tf.shape(features["inputs_raw"]))
            with tf.variable_scope(target_modality.name, reuse=None):
                features["inputs_fert"] = target_modality.bottom(features["inputs_fert_raw"])
        else:
            tf.logging.info("Fertility model NOT being used")
            features["inputs_fert_raw"] = features["inputs_raw"]
            features["inputs_fert"] = features["inputs"]
            
        

        original_targets  = features["targets"]
        inputs = features.get("inputs", None)
        hparams = self._hparams

        encoder_output, encoder_decoder_attention_bias = (None, None)
        if inputs is not None:
#            inputs, features["targets"] = common_layers.pad_to_same_length(inputs, targets)
            targets = inputs 
            target_space = features["target_space_id"]
            
            encoder_output, encoder_decoder_attention_bias = self.encode(inputs, target_space, hparams)

#            train = hparams.mode == tf.estimator.ModeKeys.TRAIN
#            targets = fertility_model(inputs, hparams, self._problem_hparams.target_modality, train, "fertility_model")
        else:
            targets = tf.zeros_like(targets)

        inputs = common_layers.flatten4d3d(features["inputs_fert"])
        
        decoder_self_attention_bias = (
            common_attention.attention_bias_lower_triangle(tf.shape(inputs)[1]))
        decoder_input = common_attention.add_timing_signal_1d(tf.nn.dropout(inputs, 0.995))

        decode_out = self.decode(decoder_input, encoder_output,
                                 encoder_decoder_attention_bias,
                                 tf.ones_like(decoder_self_attention_bias), hparams)

        with tf.variable_scope(tf.VariableScope(True)):
            with tf.variable_scope(target_modality.name + "/shared", reuse=True):
                embeddings = target_modality._get_weights()

        trans_embeddings = tf.get_variable("trans_embed", shape=embeddings.get_shape().as_list())

        discrim_features = tf.gather(trans_embeddings, features["targets_raw"])
        discrim_features = tf.reshape(discrim_features, tf.shape(original_targets))

        projected_truth = tf.matmul(tf.reshape(discrim_features, [-1, 512]),tf.stop_gradient(embeddings), transpose_b=True)
#        _, discrim_features = common_layers.pad_to_same_length(inputs, discrim_features)

        trans_embed_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
                labels=tf.reshape(features["targets_raw"], [-1]),
                logits=projected_truth,
                name="trans_embed_loss"
            )

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

        d_fake = discriminator(decode_out, features["inputs"], usage="fake", hparams=self._hparams, grad_pen=gradient_penalty, reuse=True)
        tf.summary.scalar("gen_score", tf.reduce_mean(d_fake))
        d_loss = tf.reduce_mean(d_fake - d_real)
        d_real_cycled = discriminator(tf.concat((discrim_features[1:], discrim_features[:1]), axis=0), features["inputs"], usage="real", hparams=self._hparams, reuse=True)
        d_loss_cycle = tf.reduce_mean(d_real_cycled - d_real)

        if hparams.ganmode == "wgan":
            d_vars = [var for var in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES) if "discriminator" in var.name]
            clip_ops = []
            for var in d_vars:
                clip_bounds = [-.001, .001]
                clip_ops.append(
                    tf.assign(
                        var,
                        tf.clip_by_value(var, clip_bounds[0], clip_bounds[1])
                    )
                )
            with tf.control_dependencies(clip_ops):
                d_loss = tf.identity(d_loss)
                 
        losses = {
            "discriminator": d_loss,
            "lipschitz-penalty": gradient_penalty,
            "trans_embed_loss": trans_embed_loss,
            "semantic_reg":d_loss_cycle
        }

        return decode_out, losses

    def model_fn(self, features, **kwargs):
        features["inputs"], features["targets"] = common_layers.pad_to_same_length(features["inputs"], features["targets"])
#        fert_filename = self._hparams.fertility_filename
#        if fert_filename is not None:
#            tf.logging.info("Loading Fertility Model")
#            fert_model = FertilityModel(fert_filename)
#            features["inputs_fert"] = tf.py_func(fert_model.fertilize, [features["inputs"]], tf.int32)
#            features["inputs_fert"] =  tf.reshape(features["inputs_fert"], tf.shape(features["inputs"]))
#        else:
#            tf.logging.info("Fertility model NOT being used")
#            features["inputs_fert"] = features["inputs"]
            
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
        if top_beams * beam_size != 1:
            raise NotImplementedError("top beams and beam size must be 1")
        modality = self._problem_hparams.target_modality
        raw_inputs = inputs = features["inputs"]
        inputs = tf.expand_dims(inputs, axis=1)
        if len(inputs.shape) < 5:
            inputs = tf.expand_dims(inputs, axis=4)
        s = tf.shape(inputs)
        inputs = tf.reshape(inputs, [s[0] * s[1], s[2], s[3], s[4]])

        batch_size = tf.shape(inputs)[0]
        decode_length = tf.shape(inputs)[1] + decode_length
        initial_ids = tf.zeros([batch_size, decode_length, 1, modality._body_input_depth], dtype=tf.float32)



        features["targets"] = initial_ids
        features["targets_raw"] = tf.zeros([batch_size, decode_length, 1, 1], dtype=tf.int32)

        features["targets_raw"], inputs = common_layers.pad_to_same_length(features["targets_raw"], inputs)

        
        with tf.variable_scope(modality.name, reuse=None):
            features["inputs"] = modality.bottom(inputs)

        with tf.variable_scope("body", reuse=None):
            body_out = self.model_fn_body(features)

        with tf.variable_scope(modality.name, reuse=None):
            logits = modality.top(*body_out)

        # logits = tf.squeeze(logits, -2)
        features["inputs"] = raw_inputs
        return common_layers.sample_with_temperature(logits, 0.0), None


@registry.register_hparams
def transformer_gan_base():
    hparams = transformer_base_single_gpu()
    hparams.input_modalities = "inputs:symbol:GAN"
    hparams.target_modality = "symbol:GAN"
    hparams.batch_size = 1024
    hparams.learning_rate = 1e-5
    hparams.learning_rate_decay_scheme = "none"
    hparams.optimizer = "RMSProp"
    hparams.summarize_grads = True
    hparams.clip_grad_norm = 1000.0
    hparams.num_decoder_layers = 7
    hparams.num_encoder_layers = 6
    hparams.max_length = 128
    hparams.layer_prepostprocess_dropout = 0.01
    hparams.attention_dropout = 0.01
    hparams.relu_dropout = 0.01
    hparams.add_hparam("ganmode", "wgan")
    hparams.add_hparam("num_compress_steps", 2)
    hparams.add_hparam("num_decode_steps", 0)
    hparams.add_hparam("discrim_grad_mul", 1e-10)
    hparams.add_hparam("step_interval", 1)
    hparams.add_hparam("warmup_steps", 1000001)
    hparams.add_hparam("mle_decay_period", 100000)
    hparams.add_hparam("lipschitz_mult", 1500.0)
    hparams.add_hparam("fertility_cells", 2)
    hparams.add_hparam("z_temp", 0.05)
    hparams.add_hparam("discrim_dropout", 0.01)
    hparams.add_hparam("embedding_file", "embeddings.pkl")
    hparams.add_hparam("fertility_filename", "ENG_FR.alignfertility_model.pkl")
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
            initialiser = lambda name: tf.random_normal_initializer(0.0, hidden_dim**-0.5)
        else:
            with open(self._model_hparams.embedding_file, "rb") as fp:
                embeddings = pickle.load(fp)
            tf.logging.info("Loading embeddings from file")
            initialiser = lambda name: tf.constant(embeddings["symbol_modality_27927_512/shared/"+name+":0"])

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
                    initializer=initialiser(var_name)))
        if num_shards == 1:
            ret = shards[0]
        else:
            ret = tf.concat(shards, 0)
        ret = eu.convert_gradient_to_tensor(ret)

        if type(ret) == list:
            weights = [decay_gradient(w, 1000000, summarize=False) for w in ret]
        else:
            weights = decay_gradient(ret, 1000000, summarize=False)
        return weights
        
    @property
    def targets_weights_fn(self):
        return common_layers.weights_all

    def loss(self, *args, **kwargs):
        loss, weights = super(GANSymbolModality, self).loss(*args, weights_fn=common_layers.weights_all)
        return decay_gradient(loss, 1000000, summarize=False), weights
