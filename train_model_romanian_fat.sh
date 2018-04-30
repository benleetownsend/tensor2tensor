python3 setup.py install
nohup t2t-trainer   --data_dir=../t2t_data   --problems=translate_roen_wmt8k   --model=transformer_gan   --hparams_set=transformer_gan_fat_ro_reinforce   --output_dir=../t2t_train/final_run_romanian/final_first_romanian_fat  --train_steps=5000000 --save_checkpoints_secs=1200 --eval_run_autoregressive &
