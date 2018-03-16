export CUDA_VISIBLE_DEVICES=
python3 setup.py install
nohup t2t-trainer   --data_dir=../t2t_data   --problems=translate_enfr_wmt32k   --model=transformer_gan   --hparams_set=transformer_gan_base   --output_dir=../t2t_train/final_runs/fertility_test  --train_steps=5000000&# --save_checkpoints_secs=6000&# --eval_run_autoregressive &
