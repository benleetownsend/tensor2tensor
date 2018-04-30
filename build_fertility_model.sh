export CUDA_VISIBLE_DEVICES=
python3 setup.py install

#t2t-trainer   --data_dir=../t2t_data   --problems=translate_roen_wmt8k   --model=get_features   --hparams_set=transformer_gan_base_ro   --output_dir=../t2t_train/final_runs_romanian/features_ro  --train_steps=50000

/root/code/fast_align/fast_align/build/fast_align -i translate_roen_wmt8k.align -v -d -o -I 5 > translate_roen_wmt8k.align.alignments
python3 ./tensor2tensor/fertility_model/alignments_to_fertility.py translate_roen_wmt8k.align
