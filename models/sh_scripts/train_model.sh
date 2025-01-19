DATA_DIR="./data"
DATASET=taacf
MODEL=CNN

DEVICE="mps"
N_DEVICES=1

TRAIN_BATCH_SIZE=1024
INFER_BATCH_SIZE=1024

echo "Using dataset '$DATASET', model '$MODEL', and device '$DEVICE'..."

if [[ "$DATASET" == "pk" ]]; then
    target_col="auc_bin"
elif [[ "$DATASET" == "taacf" ]]; then
    target_col="inhibition_bin"
elif [[ "$DATASET" == "mlsmr" ]]; then
    target_col="inhibition_bin"
else
    echo "Dataset '$DATASET' not recognized."
    exit 125
fi

python py_scripts/train_model.py \
    --data_dir $DATA_DIR \
    --dataset $DATASET \
    --target_col $target_col \
    --smiles_col smiles \
    --model_name $MODEL \
    --device $DEVICE \
    --n_devices $N_DEVICES \
    --n_epochs 50 \
    --tokenizer_max_length 128 \
    --patience 10 \
    --train_batch_size $TRAIN_BATCH_SIZE \
    --infer_batch_size $INFER_BATCH_SIZE \
    --use_best_config \
    --find_threshold f1 \
    --verbose