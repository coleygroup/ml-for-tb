DATA_DIR="./data"
DATASET="pk" # Options: 'pk' (190), 'taacf' (100k+), 'mlsmr' (270k+)

DEVICE="mps" # Options: 'cuda', 'cpu'
N_DEVICES=1

TRAIN_BATCH_SIZE=2048
INFER_BATCH_SIZE=2048

echo "Using dataset '$DATASET' and device '$DEVICE'..."

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


for model in FCNN CNN LSTM ChemBERTa-v2; do
    echo "Model: '$model'."
    python py_scripts/optimize_model.py \
        --data_dir $DATA_DIR \
        --dataset $DATASET \
        --target_col $target_col \
        --smiles_col smiles \
        --n_trials 10 \
        --n_warmup_steps 5 \
        --optuna_verbosity 0 \
        --model_name $model \
        --device $DEVICE \
        --n_devices $N_DEVICES \
        --n_epochs 50 \
        --tokenizer_max_length 128 \
        --patience 10 \
        --train_batch_size $TRAIN_BATCH_SIZE \
        --infer_batch_size $INFER_BATCH_SIZE \
        --save_model
done

echo "Done."