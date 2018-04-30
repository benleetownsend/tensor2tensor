from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensor2tensor.layers import common_layers
from tensor2tensor.layers import common_attention
from tensor2tensor.utils import registry
from tensor2tensor.utils.gan_utils import clip_op, critic, gen_transpose_embeddings, randomly_fertilise
from tensor2tensor.models.transformer import Transformer, transformer_base_single_gpu

from tensor2tensor.models.fertility import FertilityModel as DQNFertility
from tensor2tensor.models.language.output_smooth import SmoothOutput
import tensorflow as tf


@registry.register_model
class TransformerGAN(Transformer):
  def model_fn_body(self, features):

    # Collect modalities and features
    target_modality = self._problem_hparams.target_modality
    fert_filename = self._hparams.fertility_filename
    original_targets = features["targets"]
    inputs = features.get("inputs", None)
    hparams = self._hparams

    # Run the encoder
    encoder_output, encoder_decoder_attention_bias = self.encode(inputs, features["target_space_id"], hparams)

    # Run the fertility model if fertility is not set. Eg, ground truth fertility or at decoding.
    if features.get("inputs_fert_raw", None) is None:
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
    else:
      # A dummy reinforcement loss op.
      def reinforce(reward):
        return tf.constant(0.0)

    # Embed the fertilised inputs
    with tf.variable_scope(tf.VariableScope(True)):  # resets the variable scope to "/"
      with tf.variable_scope(target_modality.name, reuse=True):
        features["inputs_fert"] = target_modality.bottom(features["inputs_fert_raw"])

    # Flatten the inputs, add timing signal and run decoder.
    inputs = common_layers.flatten4d3d(features["inputs_fert"])

    decoder_self_attention_bias = (
      common_attention.attention_bias_lower_triangle(tf.shape(inputs)[1]))
    decoder_input = common_attention.add_timing_signal_1d(tf.nn.dropout(inputs, 1 - hparams.z_temp))

    decode_out = self.decode(decoder_input, encoder_output,
                             encoder_decoder_attention_bias,
                             tf.ones_like(decoder_self_attention_bias), hparams)

    # Get the embeddings from the modality
    with tf.variable_scope(tf.VariableScope(True)):
      with tf.variable_scope(target_modality.name + "/shared", reuse=True):
        embeddings = target_modality._get_weights()

    trans_embed_loss, targets_trans_embed = gen_transpose_embeddings(features["targets_raw"], original_targets,
                                                                     embeddings,
                                                                     hparams)

    # Score the real data with the critic.
    d_real = critic(targets_trans_embed, features["inputs"], usage="real", hparams=self._hparams)
    tf.summary.scalar("real_score", tf.reduce_mean(d_real))

    # WGAN-GP requires another pass of the critic to calculate gradients.
    gradient_penalty = 0.0
    if hparams.ganmode == "wgan-gp":
      alpha = tf.random_uniform(shape=[tf.shape(targets_trans_embed)[0], 1, 1, 1], minval=0., maxval=1.)

      differences = tf.stop_gradient(decode_out - targets_trans_embed)
      interpolates = tf.stop_gradient(targets_trans_embed) + (alpha * differences)
      gradients = tf.gradients(
        critic(interpolates, features["inputs"], usage="gp", hparams=self._hparams, reuse=True),
        [interpolates])[0]

      slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), reduction_indices=[1, 2, 3]))
      gradient_penalty = tf.reduce_mean((slopes - 1.) ** 2) * hparams.lipschitz_mult

    # The fake data critic values. The critic will automatically reverse the gradients and deal with the GAN, bits.
    d_fake = critic(decode_out, features["inputs"], usage="fake", hparams=self._hparams, reuse=True)
    tf.summary.scalar("gen_score", tf.reduce_mean(d_fake))

    # Calculate W-GAN loss (super simple eh)
    d_loss = tf.reduce_mean(d_fake - d_real)

    # The Semantic critic Loss
    d_real_cycled = critic(tf.random_shuffle(targets_trans_embed), features["inputs"], usage="real",
                           hparams=self._hparams, reuse=True)
    d_loss_cycle = tf.reduce_mean(d_real_cycled - d_real)

    # If regular WGAN, apply weight clipping to the critic values
    if hparams.ganmode == "wgan":
      clip_ops = clip_op("discriminator")
      with tf.control_dependencies(clip_ops):
        d_loss = tf.identity(d_loss)

    losses = {
      "discriminator": d_loss,
      "lipschitz-penalty": gradient_penalty,
      "trans_embed_loss": trans_embed_loss,
      "semantic_reg": d_loss_cycle * 150,
      "reinforce_fert": tf.reduce_mean(reinforce(d_fake * hparams.reinforce_delta))
    }

    # If the model is being predicted with, ignore most of the losses and just return the discriminator loss.
    if self._hparams.mode in [tf.estimator.ModeKeys.PREDICT, tf.estimator.ModeKeys.EVAL]:
      return decode_out, d_fake

    return decode_out, losses

  def model_fn(self, features, **kwargs):
    # Pads the values to the same length before theyre embedded by the model_fn.
    features["inputs"], features["targets"] = common_layers.pad_to_same_length(features["inputs"],
                                                                               features["targets"])
    return super(TransformerGAN, self).model_fn(features, **kwargs)

  def eval_autoregressive(self,
                          features=None,
                          decode_length=50):
    """
    As we dont have an autoregressive mode, the autoregressive mode is used to eval with the lang model smoothing.
    A slightly hacky way to get the model to use the lang model smoother when the model is in autoregressive mode.
    """
    raw_targets = features["targets"]
    if self._hparams.ar_beams == 1:
      return super(TransformerGAN, self).eval_autoregressive(features, decode_length)

    vocab_sz = self._problem_hparams.target_modality._vocab_size
    ids = self._fast_decode(features, decode_length,
                            beam_size=self._hparams.ar_beams,
                            top_beams=1,
                            alpha=self._hparams.ar_alpha)[0]

    fake_logits = tf.one_hot(ids, vocab_sz)
    features["targets"] = raw_targets
    return fake_logits, dict()

  def _fast_decode(self,
                   features,
                   decode_length,
                   beam_size=1,
                   top_beams=1,
                   alpha=1.0):
    """
    The decoder to get values with unknown ground truth.
    Included in here is code to use a number of attempted decoding methods.

    * Possibility to take beam_size values from the manifold, run the discriminator on them and pick the best one.
    * Possibility to smooth wrt a language model. (Used in the report)

    """
    assert beam_size == 1, "The method of trying to run the generator beam_size times is really bad, dissabled for now"

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

    features["inputs_fert_raw"] = randomly_fertilise(features["inputs_raw"])

    batch_size = tf.shape(inputs)[0]
    decode_length = tf.shape(inputs)[1] + decode_length
    initial_ids = tf.zeros([batch_size, decode_length, 1, modality._body_input_depth], dtype=tf.float32)

    features["targets"] = initial_ids
    features["targets_raw"] = tf.zeros([batch_size, decode_length, 1, 1], dtype=tf.int32)

    features["targets_raw"], inputs = common_layers.pad_to_same_length(features["targets_raw"], inputs)

    if self._hparams.lang_model_file is not None:
      smooth_decoder = SmoothOutput(self._hparams.lang_model_file, self._hparams.lang_model_data,
                                    self._hparams.problems[0].vocabulary["targets"])
      decode_fn = smooth_decoder.decode_sequence
    else:
      def decode_fn(sequence_logits):
        return common_layers.sample_with_temperature(sequence_logits, 0.0)  # argmax decoding

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

    body_out = body_out[0], None
    with tf.variable_scope(modality.name, reuse=None):
      logits = modality.top(*body_out)

    features["inputs"] = raw_inputs

    return decode_fn(logits), None


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
  hparams.add_hparam("ar_beams", 1)
  hparams.add_hparam("ar_alpha", 0.0)
  hparams.add_hparam("reinit", False)
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
  hparams.add_hparam("lang_model_file", "/root/code/t2t_data/lang_model_large.eng.pkl")
  hparams.add_hparam("lang_model_data", "/root/code/t2t_data/t2t_datagen/ro-en-lang_data_large.txt")
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
  """ A set of hyperparams that allow testing on a laptop. """
  hparams = transformer_gan_base()
  hparams.hidden_size = 128
  return hparams
