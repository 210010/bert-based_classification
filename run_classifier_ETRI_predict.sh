#! /bin/bash
BERT_BASE_DIR="/root/workspace"
#--init_checkpoint=$BERT_BASE_DIR/ETRI_morp_TF/model.ckpt-3816 \
OUTPUT_DIR="/data2/bert_record/output"
#python run_classifier_ETRI_multigpu_hook.py \
python run_classifier_ETRI_multigpu_savepb.py \
  --task_name=emot \
  --vocab_file=$BERT_BASE_DIR/ETRI_morp_TF/vocab.korean_morp.list \
  --bert_config_file=$BERT_BASE_DIR/ETRI_morp_TF/bert_config.json \
  --do_predict=True \
  --data_dir=$BERT_BASE_DIR/ETRI_morp_TF/emo_dataset \
  --do_lower_case=False \
  --max_seq_length=128 \
  --output_dir=$OUTPUT_DIR/emo_model_epoch3_withhook_clsatt \
  --use_tpu=False  
