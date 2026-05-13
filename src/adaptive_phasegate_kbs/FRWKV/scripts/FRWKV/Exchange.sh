

model_name=FRWKV

# Basic parameters (based on original Exchange script)
seq_lens=(96 96 96 96)
pred_lens=(96 192 336 720)
train_ratios=(1.0 1.0 1.0 1.0)

d_models=(256 128 256 256)
embed_size=(16 16 16 16)
cuda_ids1=(0 0 0 0)

learning_rate=(1e-4 1e-4 1e-4 1e-4)
dropout=(0.0 0.0 0.0 0.0)
#train_epochs=(30 30 30 30)
train_epochs=(4 2 1 5)


# Random seed
seed=2025

# Exchange dataset specific parameters
enc_in=8                     # Input dimension for Exchange dataset
dec_in=8                     # Decoder input dimension for Exchange dataset
c_out=8                      # Output dimension for Exchange dataset
e_layers=2                   # Number of encoder layers

# ==========================================
# Script execution section
# ==========================================

echo "=========================================="
echo "FRWKV Exchange Dataset Multi-prediction Length Experiment"
echo "=========================================="
echo "Model: ${model_name}"
echo "Dataset: Exchange Rate"
echo "Input sequence lengths: ${seq_lens[@]}"
echo "Prediction length list: ${pred_lens[@]}"
echo "Random seed: ${seed}"
echo "Model dimensions: ${d_models[@]}"
echo "Learning rates: ${learning_rate[@]}"
echo "Training epochs: ${train_epochs[@]}"
echo "Number of encoder layers: ${e_layers}"
echo "Input dimension: ${enc_in}"
echo "Total experiments: ${#pred_lens[@]}"
echo "=========================================="
echo ""

# Record start time
start_time=$(date)
echo "Experiment start time: ${start_time}"
echo ""

# Experiment counter
experiment_count=0
total_experiments=${#pred_lens[@]}

# Loop through different prediction length experiments
for ((i = 0; i < ${#pred_lens[@]}; i++)); do
    seq_len=${seq_lens[i]}
    pred_len=${pred_lens[i]}
    d_model=${d_models[i]}
    embed_size_val=${embed_size[i]}
    cuda_id=${cuda_ids1[i]}
    lr=${learning_rate[i]}
    drop=${dropout[i]}
    epochs=${train_epochs[i]}
    
    experiment_count=$((experiment_count + 1))
    
    echo "=========================================="
    echo "Experiment ${experiment_count}/${total_experiments}: Prediction length = ${pred_len}, Model dimension = ${d_model}"
    echo "=========================================="
        
        # Set GPU
        export CUDA_VISIBLE_DEVICES=${cuda_id}
        
        # Run experiment
        python -u run.py \
          --is_training 1 \
          --root_path ./dataset/exchange_rate/ \
          --data_path exchange_rate.csv \
          --model_id Exchange_frwkv_${seq_len}_${pred_len}_seed${seed} \
          --model $model_name \
          --data custom \
          --features M \
          --seq_len ${seq_len} \
          --pred_len ${pred_len} \
          --enc_in ${enc_in} \
          --dec_in ${dec_in} \
          --c_out ${c_out} \
          --des 'Exp' \
          --embed_size ${embed_size_val} \
          --d_model ${d_model} \
          --d_ff ${d_model} \
          --batch_size 32 \
          --learning_rate ${lr} \
          --itr 1 \
          --e_layers ${e_layers} \
          --lossfun_alpha 0.5 \
          --test_batch_size 16 \
          --test_mode 0 \
          --CKA_flag 0 \
          --fix_seed 1 \
          --seed ${seed} \
          --resume_training 0 \
          --resume_epoch 0 \
          --save_every_epoch 0 \
          --use_revin 1 \
          --use_norm 1 \
          --send_mail 0 \
          --save_pdf 0 \
          --train_epochs ${epochs} \
          --patience 8 \
          --lradj type1 \
          --loss_mode L1 \
          --train_ratio 1.0 \
          --dropout ${drop} \
          --plot_mat_flag 0 \
          2>&1 | tee -a logs/frwkv/FRWKV_Exchange_${seq_len}_${pred_len}_seed${seed}.log
        
    # Check if experiment completed successfully
    if [ $? -eq 0 ]; then
        echo "✅ Experiment for prediction length ${pred_len}, model dimension ${d_model} completed successfully"
    else
        echo "❌ Experiment for prediction length ${pred_len}, model dimension ${d_model} failed"
    fi
    
    echo "=========================================="
    echo ""
    
    # If not the last experiment, take a break
    if [ $experiment_count -lt $total_experiments ]; then
        echo "Resting for 1 second before continuing to next experiment..."
        sleep 1
        echo ""
    fi
done

# Record end time
end_time=$(date)
echo "=========================================="
echo "All experiments completed!"
echo "=========================================="
echo "Start time: ${start_time}"
echo "End time: ${end_time}"
echo "Total experiments: ${total_experiments}"
echo "Experiment results saved in: logs/frwkv/"
echo "=========================================="

# Generate result summary
echo "=========================================="
echo "Experiment result summary"
echo "=========================================="
echo "Experiment configuration:"
echo "- Model: ${model_name}"
echo "- Dataset: Exchange Rate"
echo "- Sequence lengths: ${seq_lens[@]}"
echo "- Prediction lengths: ${pred_lens[@]}"
echo "- Random seed: ${seed}"
echo "- Total experiments: ${total_experiments}"
echo "- Input dimension: ${enc_in}"
echo "- Number of encoder layers: ${e_layers}"
echo "- Model dimension configuration: ${d_models[@]}"
echo ""
echo "Log file list:"
for ((i = 0; i < ${#pred_lens[@]}; i++)); do
    seq_len=${seq_lens[i]}
    pred_len=${pred_lens[i]}
    log_file="logs/frwkv/FRWKV_Exchange_${seq_len}_${pred_len}_seed${seed}.log"
    if [ -f "$log_file" ]; then
        echo "✅ ${log_file}"
    else
        echo "❌ ${log_file} (not found)"
    fi
done
echo "=========================================="

# Generate experiment statistics
echo "=========================================="
echo "Experiment statistics"
echo "=========================================="
echo "Prediction length statistics:"
for ((i = 0; i < ${#pred_lens[@]}; i++)); do
    pred_len=${pred_lens[i]}
    log_file="logs/frwkv/FRWKV_Exchange_${seq_lens[i]}_${pred_len}_seed${seed}.log"
    if [ -f "$log_file" ]; then
        echo "- Prediction length ${pred_len}: ✅ Completed"
    else
        echo "- Prediction length ${pred_len}: ❌ Not completed"
    fi
done
echo ""
echo "Experiment completion statistics:"
log_file="logs/frwkv/FRWKV_Exchange_*_*_seed${seed}.log"
if ls $log_file 1> /dev/null 2>&1; then
    completed_count=$(ls $log_file | wc -l)
    echo "- Total completed: ${completed_count}/${#pred_lens[@]} experiments"
else
    echo "- Total completed: 0/${#pred_lens[@]} experiments"
fi
echo "=========================================="

# Generate detailed statistics report
echo "=========================================="
echo "Detailed statistics report"
echo "=========================================="
echo "Dataset: Exchange Rate"
echo "Prediction length completion status:"
for ((i = 0; i < ${#pred_lens[@]}; i++)); do
    seq_len=${seq_lens[i]}
    pred_len=${pred_lens[i]}
    d_model=${d_models[i]}
    log_file="logs/frwkv/FRWKV_Exchange_${seq_len}_${pred_len}_seed${seed}.log"
    if [ -f "$log_file" ]; then
        echo "  - Prediction length ${pred_len} (model dimension ${d_model}): ✅ Completed"
    else
        echo "  - Prediction length ${pred_len} (model dimension ${d_model}): ❌ Not completed"
    fi
done
echo ""
echo "Experiment parameters summary:"
echo "- Model: ${model_name}"
echo "- Dataset path: ./dataset/exchange_rate/"
echo "- Data file: exchange_rate.csv"
echo "- Input dimension: ${enc_in}"
echo "- Number of encoder layers: ${e_layers}"
echo "- Learning rates: ${learning_rate[@]}"
echo "- Dropout rates: ${dropout[@]}"
echo "- Batch size: 32"
echo "- Early stopping patience: 8"
echo "- Learning rate scheduler: type1"
echo "- Test mode: 0"
echo "- Save PDF: No"
echo "- Plot flag: No"
echo "=========================================="
