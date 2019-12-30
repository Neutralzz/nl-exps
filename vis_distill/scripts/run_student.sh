export MNLI_DATA_DIR=/mnt/unilm/hanbao/exp/mnli/MNLI
export TEACHER_MODEL_DIR=/mnt/unilm/hanbao/exp/bert_kd/mnli_teacher2
export OUTPUT_DIR=/mnt/v-zhli7/tmp/mnli_student_test

python run_glue_for_distill.py  \
    --data_dir $MNLI_DATA_DIR \
    --model_type bert \
    --model_name_or_path bert-base-uncased \
    --task_name MNLI \
    --output_dir $OUTPUT_DIR \
    --teacher_model_dir $TEACHER_MODEL_DIR \
    --student_config student_config.json \
    --teacher_config teacher_config.json \
    --distill_hidden \
    --distill_attention \
    --distill_predict \
    --lambda_kd_h 1 \
    --lambda_kd_a 1 \
    --lambda_kd_p 1 \
    --lambda_kd_T 5 \
    --do_train --do_eval --do_lower_case \
    --max_seq_length 128 \
    --per_gpu_train_batch_size 32 \
    --learning_rate 5e-5 \
    --num_train_epochs 1 \
    --logging_steps 50 \
    --save_steps 2000 \
    --eval_all_checkpoints \
    --overwrite_output_dir \
    --attention_comb \
    --attention_comb_init avg \
    --comb_weights_softmax false
    