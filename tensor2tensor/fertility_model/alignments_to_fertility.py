import sys
import os
import numpy as np
import pickle as pkl
from tqdm import tqdm 
import codecs

class FertilityModel:
    def __init__(self, pickle, delta_reinforce):
        self.pickle = pickle
        with open(pickle, "rb") as fp:
            self.probs = pkl.load(fp)
        self.max_fert = self.probs.shape[1]
        self.delta_reinforce = delta_reinforce

        self.probs[self.probs<=0] = 0.0
        counts_sum = np.sum(self.probs, 1)
        self.probs = self.probs/counts_sum[:,None]
        
    def sample(self,token_id):
        return np.random.choice(self.max_fert, p=self.probs[token_id])
    
    def fertilize(self, inputs):
        outputs = np.zeros_like(inputs)
        fertilities = np.zeros_like(inputs)
#        print(outputs.shape)
        inputs_squeezed = np.squeeze(inputs, axis=(2,3))
        for batch_id, sentence in enumerate(inputs_squeezed):
            current_idx = 0
            for token_idx, token in enumerate(sentence):
                fertility = self.sample(token)
                outputs[batch_id, current_idx:current_idx+fertility, 0, 0] = token
                current_idx += fertility
                fertilities[batch_id, token_idx, 0, 0] = fertility
        return outputs, fertilities

    def reinforce(self, inputs, fertilities, losses):
#        print("inputs: ", inputs.shape)
#        print("fertilities :", fertilities.shape)
#        print("losses :", losses.shape)
        if self.delta_reinforce != 0.0:
            avg_loss = np.median(losses)
            deviations = losses - avg_loss
            fertilities = np.squeeze(fertilities, axis=(2,3))
#            fertilities = np.zeros_like(inputs)
            inputs_squeezed = np.squeeze(inputs, axis=(2,3))
            for batch_id, (sentence, ferts, dev) in enumerate(zip(inputs_squeezed, fertilities, deviations)):
                for token_idx, (token, f)  in enumerate(zip(sentence, ferts)):
                    self.probs[token, f] += (dev*self.delta_reinforce)

            self.probs[self.probs<=self.delta_reinforce] = self.delta_reinforce

            counts_sum = np.sum(self.probs, 1)
            self.probs = self.probs/counts_sum[:,None]
            with open(self.pickle+"temp", "wb") as fp:
                pkl.dump(self.probs, fp)
            print(self.probs[1])
            os.rename(self.pickle+"temp", self.pickle)
        return np.int(0)

if __name__ == "__main__":
    input_sentences = sys.argv[1]
    input_alignments = sys.argv[1]+".alignments"
#    sentences = open(input_sentences, "r")
#    alignments = open(input_alignments, "r")

    counts = np.zeros([32000, 5], dtype=np.float32) + 1

#    for sent, align in tqdm(zip(sentences, alignments)):
#        try:
#            tokenized = sent.split()
#            limit = tokenized.index("|||")
#            for i, token in enumerate(tokenized[:limit]):
#                counts[int(token), align.count(" "+str(i)+"-")] += 1
#        except:
#            pass

    counts_sum = np.sum(counts, 1)
            
    probs = counts/counts_sum[:,None]

    with open(input_sentences+"fertility_model.pkl", "wb") as fp:
        pkl.dump(probs, fp)

    fert_mod = FertilityModel(input_sentences+"fertility_model.pkl", 0.1)
    print(fert_mod.fertilize(np.reshape([1,2,3,4,5,6,7,0,0,0], [1,10,1,1])))

    



    
        
