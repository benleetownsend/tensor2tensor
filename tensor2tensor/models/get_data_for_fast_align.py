from tensor2tensor.models.transformer_gan import TransformerGAN
import tensorflow as tf
import numpy as np
from tensor2tensor.utils import registry

fp = open("translate_roen_wmt8k.align", "wt")

def write_to_file(inputs, targets):
    for inp, tar in zip(inputs, targets):
        string =  " ".join([str(i[0,0]) for i in list(inp) if i > 0]) + " ||| " + " ".join([str(i[0,0]) for i in list(tar) if i > 0])+"\n"
        fp.write(string)
    return np.float32(0)
    
@registry.register_model
class GetFeatures(TransformerGAN):
    def model_fn_body(self, features):
        targets = features["targets_raw"]
        inputs = features["inputs_raw"]
        write = tf.py_func(write_to_file, [inputs, targets], tf.float32)
        mult = tf.get_variable("noise_bandwidth", dtype=tf.float32, shape=[512])
        with tf.control_dependencies([write]):
            return features["targets"] * mult


        

