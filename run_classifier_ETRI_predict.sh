#! /bin/bash
BERT_BASE_DIR="/root/workspace"
#--init_checkpoint=$BERT_BASE_DIR/ETRI_morp_TF/model.ckpt \
OUTPUT_DIR="/data2/bert_record/output"
python run_classifier_ETRI_multigpu.py \
  --task_name=emot \
  --vocab_file=$BERT_BASE_DIR/ETRI_morp_TF/vocab.korean_morp.list \
  --bert_config_file=$BERT_BASE_DIR/ETRI_morp_TF/bert_config.json \
  --init_checkpoint=$OUTPUT_DIR/emo_model/model.ckpt-3816 \
  --do_predict=True \
  --data_dir=$BERT_BASE_DIR/ETRI_morp_TF/emo_dataset \
  --do_lower_case=False \
  --max_seq_length=128 \
  --output_dir=$OUTPUT_DIR/emo_model/prediction \
  --use_tpu=False  
