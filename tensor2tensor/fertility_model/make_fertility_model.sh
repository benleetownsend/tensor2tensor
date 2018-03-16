/root/code/fast_align/fast_align/build/fast_align -i $1 -v -d -o -I 30 > $1.alignments 
python3 ./tensor2tensor/fertility_model/alignments_to_fertility.py $1


