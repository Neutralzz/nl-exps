export GLUE_DIR=/home/aistudio/data/glue
export TASK_NAME=MNLI
export OUTPUT_DIR=/home/aistudio/work/output/test

python run_glue.py \
    --model_type bert \
    --model_name_or_path bert-base-uncased \
    --task_name $TASK_NAME \
    --config_name ./config.json \
    --do_train \
    --do_eval \
    --do_lower_case \
    --data_dir $GLUE_DIR/$TASK_NAME \
    --max_seq_length 128 \
    --per_gpu_eval_batch_size=16   \
    --per_gpu_train_batch_size=32   \
    --save_steps 0 \
    --learning_rate 2e-5 \
    --num_train_epochs 3.0 \
    --output_dir $OUTPUT_DIR