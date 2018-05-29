python3 setup.py install
nohup t2t-trainer   --data_dir=../t2t_data   --problems=translate_ende_wmt8k   --model=transformer_gan   --hparams_set=transformer_gan_german   --output_dir=../t2t_train/final_run_romanian/final_first_german_for_demo_extra_steps  --train_steps=5000000 --save_checkpoints_secs=1200&# --eval_run_autoregressive &
