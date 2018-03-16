export CUDA_VISIBLE_DEVICES=
python3 setup.py install
nohup t2t-trainer   --data_dir=../t2t_data   --problems=translate_enfr_wmt32k   --model=get_features   --hparams_set=transformer_gan_base   --output_dir=../t2t_train/final_runs/features  --train_steps=5000000 &> feats_nohup.out & 
