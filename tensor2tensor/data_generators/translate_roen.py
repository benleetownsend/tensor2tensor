# coding=utf-8
# Copyright 2017 The Tensor2Tensor Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Data generators for translation data-sets."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Dependency imports

from tensor2tensor.data_generators import generator_utils
from tensor2tensor.data_generators import problem
from tensor2tensor.data_generators import text_encoder
from tensor2tensor.data_generators import translate
from tensor2tensor.utils import registry

import tensorflow as tf

FLAGS = tf.flags.FLAGS

# End-of-sentence marker.
EOS = text_encoder.EOS_ID

_ROEN_TRAIN_SMALL_DATA = [
    [
    ],
]
_ROEN_TEST_SMALL_DATA = [
    [
    ],
]
_ROEN_TRAIN_LARGE_DATA = [
    [
        "http://data.statmt.org/wmt16/translation-task/training-parallel-ep-v8.tgz",
        ("training-parallel-ep-v8/europarl-v8.ro-en.ro", "training-parallel-ep-v8/europarl-v8.ro-en.en")
    ],
    [
        "http://opus.nlpl.eu/download.php?f=SETIMES2/en-ro.txt.zip",
        ("SETIMES2.en-ro.ro","SETIMES2.en-ro.en")
    ],
]
_ROEN_TEST_LARGE_DATA = [
    [
        "http://data.statmt.org/wmt17/translation-task/dev.tgz",
        ("dev/newstest2016-roen-src.ro.sgm", "dev/newstest2016-roen-ref.en.sgm")
    ],
]


@registry.register_problem
class TranslateRoenWmt8k(translate.TranslateProblem):
  """Problem spec for WMT Ro-En translation."""

  @property
  def targeted_vocab_size(self):
    return 2**13  # 8192

  @property
  def vocab_name(self):
    return "vocab.roen"

  @property
  def use_small_dataset(self):
    return True

  def generator(self, data_dir, tmp_dir, train):
    symbolizer_vocab = generator_utils.get_or_generate_vocab(
        data_dir, tmp_dir, self.vocab_file, self.targeted_vocab_size,
        _ROEN_TRAIN_LARGE_DATA)
    datasets = _ROEN_TRAIN_LARGE_DATA if train else _ROEN_TEST_LARGE_DATA
    tag = "train" if train else "dev"
    data_path = translate.compile_data(tmp_dir, datasets,
                                       "wmt_roen_tok_%s" % tag)
    return translate.token_generator(data_path + ".lang1", data_path + ".lang2",
                                     symbolizer_vocab, EOS)

  @property
  def input_space_id(self):
    return problem.SpaceID.RO_TOK

  @property
  def target_space_id(self):
    return problem.SpaceID.EN_TOK

@registry.register_problem
class TranslateRoenWmt32k(TranslateRoenWmt8k):

  @property
  def targeted_vocab_size(self):
    return 2**15  # 32768
