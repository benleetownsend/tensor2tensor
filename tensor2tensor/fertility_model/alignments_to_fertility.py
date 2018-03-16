import sys
import numpy as np
import pickle as pkl
from tqdm import tqdm 
import codecs

class FertilityModel:
    def __init__(self, pickle):
        with open(pickle, "rb") as fp:
            self.probs = pkl.load(fp)
        self.max_fert = self.probs.shape[1]
        
    def sample(self,token_id):
        return np.random.choice(self.max_fert, p=self.probs[token_id])
    
    def fertilize(self, inputs):
        outputs = np.zeros_like(inputs)
        print(outputs.shape)
        inputs_squeezed = np.squeeze(inputs, axis=(2,3))
        for batch_id, sentence in enumerate(inputs_squeezed):
            current_idx = 0
            for token in sentence:
                fertility = self.sample(token)
                outputs[batch_id, current_idx:current_idx+fertility, 0, 0] = token
                current_idx += fertility
        return outputs
    

if __name__ == "__main__":
    input_sentences = sys.argv[1]
    input_alignments = sys.argv[1]+".alignments"
    sentences = open(input_sentences, "r")
    alignments = open(input_alignments, "r")

    counts = np.zeros([32000, 50], dtype=np.float32) + 1e-10

    for sent, align in tqdm(zip(sentences, alignments)):
        try:
            tokenized = sent.split()
            limit = tokenized.index("|||")
            #    print(tokenized)
            #    exit()
            for i, token in enumerate(tokenized[:limit]):
#                print(token)
                counts[int(token), align.count(str(i)+"-")] += 1
        except:
            pass

    counts_sum = np.sum(counts, 1)
            
    probs = counts/counts_sum[:,None]

    with open(input_sentences+"fertility_model.pkl", "wb") as fp:
        pkl.dump(probs, fp)

    fert_mod = FertilityModel(input_sentences+"fertility_model.pkl")
    print(fert_mod.fertilize(np.reshape([1,2,3,4,5,6,7,0,0,0], [1,10,1,1])))

    



    
        
