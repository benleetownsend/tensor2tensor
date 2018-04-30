python3 setup.py install
nohup t2t-trainer   --data_dir=../t2t_data   --problems=translate_roen_wmt8k   --model=transformer   --hparams_set=transformer_base_single_gpu   --output_dir=../t2t_train/final_run_romanian/final_first_romanian_trans  --train_steps=5000000 --eval_run_autoregressive &
