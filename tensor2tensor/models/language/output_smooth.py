from tensor2tensor.models.language.ngram import LangModel
import tensorflow as tf
import pickle as pkl
import numpy as np
import sys

_EOS = 1
_PAD = 0

epsilon = 1e-100

def beam_is_finished(beam):
    if len(beam) == 0 or beam[-1] != _EOS:
        return False
    return True

def calc_len_pen(beams, alpha, close_beams=False):
    len_pen = np.ones(len(beams))
    for bi, beam in enumerate(beams):
        length = len(set(beam))
        len_pen[bi] = ((5.0 + length + (1.0 if close_beams or beam_is_finished(beam) else 2.0))/6.0)**(alpha or 0.0)
        if len(beam) == 0 and close_beams:
            len_pen[bi] = epsilon
    return len_pen

def biased_beam_search_decoder(logits, predict_next, beam_size, alpha=5.7):
    print("Decoding Sequence")
    beams = [[]]
    log_probs = np.zeros(len(beams)) - 0.1 # to cater for the empty case.
    for t, step_logits in enumerate(logits):
        min_log_prob = np.max(step_logits) - np.std(step_logits) * 2
        num_samples_t = np.sum(step_logits > min_log_prob)
        step_logits[_PAD]  = -np.inf # pad should never be predicted as we have variable length beams that stop at EOS
        candidate_ids =  np.argsort(step_logits)[-num_samples_t:]
        samples = np.zeros((len(beams), num_samples_t))
        for bi, beam in enumerate(beams):
            if not beam_is_finished(beam):
                for ci, candidate_logit in enumerate(candidate_ids):
                    samples[bi, ci] = log_probs[bi] + predict_next(beam, candidate_logit) + step_logits[candidate_logit]
            else:
                samples[bi,:] = -np.inf
                samples[bi, 0] = log_probs[bi]

        curr_num_beams = min((beam_size, len(beams)))
        top_samples = np.unravel_index(np.argpartition((samples/calc_len_pen(beams, alpha)[:,None]).ravel(), -curr_num_beams)[-curr_num_beams:], samples.shape)
        new_beams = []
        new_log_probs  = np.ones(curr_num_beams + len(beams)) * -np.inf
        
        for i, (b_i, c_i) in enumerate(zip(*top_samples)):
            new_beams.append(beams[b_i][:]+([candidate_ids[c_i]] if not beam_is_finished(beams[b_i]) else []))
            new_log_probs[i] = samples[b_i, c_i]

        new_beams += beams[:] # add the old beams un-changed
        new_log_probs[-len(beams):]  = log_probs[:]        
        beams = new_beams
        log_probs = new_log_probs
        try:
            top_final_beams = np.argpartition(log_probs/calc_len_pen(beams, alpha, close_beams=True), -beam_size)[-beam_size:]
        except ValueError:
            # if the number of available beams is less than double the set beam size, take all of them.
            top_final_beams = np.arange(len(beams))

        beams = [b for i, b in enumerate(beams) if i in top_final_beams]
        log_probs = log_probs[top_final_beams]

    output_beams = []
    output_probs = []
    logit_len = len(logits)
    for beam, prob in zip(beams, log_probs):
        if beam_is_finished(beam):
            output_beams.append(beam)
            output_probs.append(prob)
        else:
            beam = beam[:logit_len - 1] # if the sequence is too long, trim off the last token - this is an edge case and should never happen.
            output_beams.append(beam + [_EOS])
            output_probs.append(prob + predict_next(beam, _EOS)) # Force an EOS token on the end and recalculate the probabilities

    beams = output_beams
    log_probs = np.array(output_probs)

    logit_len = len(logits)
    len_pen = calc_len_pen(beams, alpha, close_beams=True)
    beams = [b + [_PAD] * (logit_len - len(b)) for b in beams]
    return beams, log_probs/len_pen

class SmoothOutput:
    def __init__(self, filepath, data=None, tokenizer=None):
        try:
            tf.logging.info("Attempting to load language model")
            with open(filepath, "rb") as fp:
                self.lm = pkl.load(fp)
        except:
            tf.logging.info("Failed...Generating Language Model")
        
            tokenized = []
            def data_gen():
                with open(data, "rt") as fp:
                    for line, _ in zip(fp, range(20000000000000)):
                        yield tokenizer.encode(line.strip()) + [_EOS]

            self.lm = LangModel()
            self.lm.fit(data_gen)
            with open(filepath, "wb") as fp:
                pkl.dump(self.lm, fp)

        if tokenizer is None:
            def tokenizer(i):
                return i
        for i in range(10, 50):
            seq = self.lm.lm.generate(i)
            print(tokenizer.decode(seq))

    def decode_sequence_py(self, logits):
        logits = np.squeeze(logits, (2,3))
        output = []
        #logits =  batch, length, 1 , vocab_sz
        for sequences in logits:
            beams, lprob = biased_beam_search_decoder(sequences, self.lm.predict, 50)
            output.append(beams[np.argmax(lprob)])
        return np.array(output, dtype=np.int32)

    def decode_sequence(self, logits):
        log_probs = tf.nn.log_softmax(logits)
        output = tf.py_func(self.decode_sequence_py, [log_probs], [tf.int32])
        return tf.reshape(output, tf.shape(log_probs)[:-1])
            
            
if __name__ == "__main__":
    SmoothOutput("/root/code/t2t_data/lang_model.eng.pkl")
