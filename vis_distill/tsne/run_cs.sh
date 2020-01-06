export TASK_NAME=MNLI
export GLUE_DIR=/home/aistudio/data/glue_data
export OUTPUT_DIR=/home/aistudio/work/teacher_models/model-$TASK_NAME
export PYTORCH_PRETRAINED_BERT_CACHE=/home/aistudio/data/pretrained_models

python compute_score_for_teacher.py \
    --data_dir $GLUE_DIR/$TASK_NAME \
    --task_name $TASK_NAME \
    --output_dir $OUTPUT_DIR \
    --do_eval \
    --do_lower_case \
    --eval_all_checkpoints