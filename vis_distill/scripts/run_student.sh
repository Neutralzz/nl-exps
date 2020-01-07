export TASK_NAME=MNLI
export GLUE_DIR=/home/aistudio/data/glue_data
export TEACHER_MODEL_DIR=/home/aistudio/work/teacher_models/model-$TASK_NAME
export OUTPUT_DIR=/home/aistudio/work/output/run-$TASK_NAME
export PYTORCH_PRETRAINED_BERT_CACHE=/home/aistudio/data/pretrained_models

python run_glue_for_distill.py \
    --task_name $TASK_NAME \
    --data_dir $GLUE_DIR/$TASK_NAME \
    --model_type bert \
    --model_name_or_path $TEACHER_MODEL_DIR \
    --teacher_model_dir $TEACHER_MODEL_DIR \
    --student_config student_config.json \
    --teacher_config teacher_config.json \
    --output_dir $OUTPUT_DIR \
    --student_hidden_layers 4 \
    --student_weight_select 1-3-4-10 \
    --distill_hidden_select 1-3-4-10 \
    --distill_hidden \
    --distill_predict \
    --lambda_kd_h 1 \
    --lambda_kd_p 1 \
    --lambda_kd_T 5 \
    --do_train --do_eval --do_lower_case \
    --max_seq_length 128 \
    --per_gpu_train_batch_size 32 \
    --learning_rate 5e-5 \
    --num_train_epochs 5 \
    --logging_steps 500 \
    --save_steps 2000 \
    --eval_all_checkpoints