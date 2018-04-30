YEAR=2013
#YEAR=2014
#YEAR=2015
BEAM_SIZE=1
ALPHA=0.9
export CUDA_VISIBLE_DEVICES=
t2t-decoder   --data_dir=../t2t_data   --problems=translate_roen_wmt8k   \
	      --model=transformer_gan   --hparams_set=transformer_gan_fat_ro_reinforce   \
	      --output_dir=../t2t_train/final_run_romanian/final_first_romanian_fat/  --decode_hparams="beam_size=$BEAM_SIZE" \
	      --decode_interactive

#	      --decode_aelpha=$ALPHA   --decode_from_file=/tmp/t2t_datagen/dev/newstest${YEAR}.en
#	      --decode_interactive
mkdir /tmp/t2t_datagen/dev/
cp  /root/code/t2t_data/t2t_datagen/baseline-1M-enfr/baseline-1M_test.fr /tmp/t2t_datagen/dev/newstest${YEAR}.fr
cp  /root/code/t2t_data/t2t_datagen/baseline-1M-enfr/baseline-1M_test.en /tmp/t2t_datagen/dev/newstest${YEAR}.en

#Tokenize the reference
perl ~/code/mosesdecoder/scripts/tokenizer/tokenizer.perl -l fr < /tmp/t2t_datagen/dev/newstest${YEAR}.fr > /tmp/t2t_datagen/dev/newstest${YEAR}.fr.tok
#Do compound splitting on the reference
perl -ple 's{(\S)-(\S)}{$1 ##AT##-##AT## $2}g' < /tmp/t2t_datagen/dev/newstest${YEAR}.fr.tok > /tmp/t2t_datagen/dev/newstest${YEAR}.fr.atat
#wc -l /tmp/t2t_datagen/dev/newstest${YEAR}.fr.atat
#perl -ple 's{(\S)\»(\S)}{$1 &quot; $2}g' < /tmp/t2t_datagen/dev/newstest${YEAR}.fr.atat > /tmp/t2t_datagen/dev/newstest${YEAR}.fr.atat
#wc -l /tmp/t2t_datagen/dev/newstest${YEAR}.fr.atat
#perl -ple 's{(\S)«(\S)}{$1 &quot; $2}g' < /tmp/t2t_datagen/dev/newstest${YEAR}.fr.atat > /tmp/t2t_datagen/dev/newstest${YEAR}.fr.atat
#Tokenize the translation
perl ~/code/mosesdecoder/scripts/tokenizer/tokenizer.perl -l fr < /tmp/t2t_datagen/dev/newstest${YEAR}.en.transformer_gan.transformer_gan_base.translate_enfr_wmt32k.beam1.alpha0.6.decodes > /tmp/t2t_datagen/dev/newstest${YEAR}.en.transformer_gan.transformer_base.beam1.alpha0.6.tok
#Do compount splitting on the translation
perl -ple 's{(\S)-(\S)}{$1 ##AT##-##AT## $2}g' < /tmp/t2t_datagen/dev/newstest${YEAR}.en.transformer_gan.transformer_base.beam1.alpha0.6.tok > /tmp/t2t_datagen/dev/newstest${YEAR}.en.transformer_gan.transformer_base.beam1.alpha0.6.atat

#perl -ple 's{(\S)»(\S)}{$1 &quot; $2}g' < /tmp/t2t_datagen/dev/newstest${YEAR}.en.transformer_gan.transformer_base.beam5.alpha0.6.atat > /tmp/t2t_datagen/dev/newstest${YEAR}.en.transformer_gan.transformer_base.beam5.alpha0.6.atat
#perl -ple 's{(\S)«(\S)}{$1 &quot; $2}g' < /tmp/t2t_datagen/dev/newstest${YEAR}.en.transformer_gan.transformer_base.beam5.alpha0.6.atat > /tmp/t2t_datagen/dev/newstest${YEAR}.en.transformer_gan.transformer_base.beam5.alpha0.6.atat

#Score the translation
perl ~/code/mosesdecoder/scripts/generic/multi-bleu.perl  /tmp/t2t_datagen/dev/newstest${YEAR}.fr.atat < /tmp/t2t_datagen/dev/newstest${YEAR}.en.transformer_gan.transformer_base.beam1.alpha0.6.atat
