#! /bin/bash
BERT_BASE_DIR="/root/workspace"
#--init_checkpoint=$BERT_BASE_DIR/ETRI_morp_TF/model.ckpt \
OUTPUT_DIR="/data2/bert_record/output"
python run_classifier_ETRI_multigpu_hook.py \
  --task_name=emot \
  --vocab_file=$BERT_BASE_DIR/ETRI_morp_TF/vocab.korean_morp.list \
  --bert_config_file=$BERT_BASE_DIR/ETRI_morp_TF/bert_config.json \
  --init_checkpoint=$BERT_BASE_DIR/ETRI_morp_TF/model.ckpt\
  --do_train=True \
  --do_eval=True \
  --data_dir=$BERT_BASE_DIR/ETRI_morp_TF/emo_dataset \
  --do_lower_case=False \
  --train_batch_size=8 \
  --learning_rate=8.5e-5 \
  --num_train_epochs=3.0 \
  --save_checkpoints_steps=1000 \
  --max_seq_length=128 \
  --output_dir=$OUTPUT_DIR/emo_model_epoch3_withhook_clsatt2 \
  --use_tpu=False  
