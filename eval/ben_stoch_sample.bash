set -e
export CUDA_VISIBLE_DEVICES=
lang1="ro"
lang2="en"
input_file="newsdev2016-roen"
problem="translate_roen_wmt8k"
hparams="transformer_gan_fat_ro_reinforce"
BEAM_SIZE=1
data_dir="../../t2t_data"
model_dir="../../t2t_train/final_run_romanian/final_first_romanian_fat/"

#python3 ../tensor2tensor/utils/avg_checkpoints.py --num_last_checkpoints 5 --output_path "checkpoints/averaged.ckpt" --prefix ${model_dir}

#sed -e 's/<[^>]*>//g; /^\s*$/d;s/^[ \t]*//;:s' ${input_file}.src.${lang1}.sgm  > ${input_file}.src.${lang1}
#sed -e 's/<[^>]*>//g; /^\s*$/d;s/^[ \t]*//;:s' ${input_file}.ref.${lang2}.sgm  > ${input_file}.ref.${lang2}

t2t-decoder   --data_dir=${data_dir}   --problems=${problem}   \
	      --model=transformer_gan   --hparams_set=${hparams}   \
	      --output_dir=checkpoints/  --decode_hparams="beam_size=$BEAM_SIZE"\
	      --decode_interactive
#	      --decode_from_file=${input_file}.src.${lang1}\
#	      --decode_to_file=${input_file}.trans.${lang2}

cp newsdev2016-roen.trans.en.transformer_gan.transformer_gan_fat_ro_reinforce.translate_roen_wmt8k.beam1.alpha0.6.decodes ${input_file}.trans.${lang2}

#sed -e 's/<[^>]*>//g; /^\s*$/d;s/^[ \t]*//;:s;s/\(\<\S*\>\)\(.*\)\<\1\>/\1\2/g;ts;s/  */ /g' ${input_file}.trans.${lang2} > ${input_file}.trans.${lang2}

sed -i  -e 's/\(.*\)/\L\1/;:s;s/\(\<\S*\>\)\(.*\)\<\1\>/\1\2/g;ts;s/  */ /g' ${input_file}.ref.${lang2}
sed -i -e 's/\(.*\)/\L\1/;s/<pad>//g;s/  */ /g;:s;s/\(\<\S*\>\)\(.*\)\<\1\>/\1\2/g;ts;s/  */ /g' ${input_file}.trans.${lang2}

perl ~/code/mosesdecoder/scripts/tokenizer/tokenizer.perl -l ${lang2} < ${input_file}.ref.${lang2} > ref.${lang2}
perl ~/code/mosesdecoder/scripts/tokenizer/tokenizer.perl -l ${lang2} < ${input_file}.trans.${lang2} > trans.${lang2}
#~/code/mosesdecoder/scripts/tokenizer/tokenizer.perl -l $lang1 < $input_file.src.$lang1 > src.$lang1

~/code/mosesdecoder/scripts/generic/multi-bleu.perl ref.${lang2} < trans.${lang2}
