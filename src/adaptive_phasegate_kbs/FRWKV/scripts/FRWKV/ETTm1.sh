model_name=FRWKV

# Basic parameters
seq_len=96                    # Input sequence length
train_ratio=1.0              # Training data ratio
d_model=512                  # Model dimension
embed_size=16                # Embedding dimension
cuda_id=0                    # GPU ID
learning_rate=1e-4           # Learning rate
dropout=0.2                  # Dropout rate


# Prediction length list
pred_lengths=(96 192 336 720)

# Random seed list
seeds=(2020 2021 2022 2023 2024 2025 2026)

# Training parameters
batch_size=32                # Batch size
train_epochs=30              # Training epochs
patience=8                   # Early stopping patience

# ==========================================
# Script execution section
# ==========================================

export CUDA_VISIBLE_DEVICES=${cuda_id}

echo "=========================================="
echo "FRWKV Multi-prediction Length Multi-seed Experiment"
echo "=========================================="
echo "Model: ${model_name}"
echo "Input sequence length: ${seq_len}"
echo "Prediction length list: ${pred_lengths[@]}"
echo "Random seed list: ${seeds[@]}"
echo "Model dimension: ${d_model}"
echo "Learning rate: ${learning_rate}"
echo "Training epochs: ${train_epochs}"
echo "Total experiments: $((${#pred_lengths[@]} * ${#seeds[@]}))"
echo "=========================================="
echo ""

# Record start time
start_time=$(date)
echo "Experiment start time: ${start_time}"
echo ""

# Experiment counter
experiment_count=0
total_experiments=$((${#pred_lengths[@]} * ${#seeds[@]}))

# Loop through different prediction length and random seed experiments
for pred_len in "${pred_lengths[@]}"; do
    for seed in "${seeds[@]}"; do
        experiment_count=$((experiment_count + 1))
        
        echo "=========================================="
        echo "Experiment ${experiment_count}/${total_experiments}: Prediction length = ${pred_len}, Random seed = ${seed}"
        echo "=========================================="
        
        # Run experiment
        python -u run.py \
          --is_training 1 \
          --root_path ./dataset/ETT-small/ \
          --data_path ETTm1.csv \
          --model_id ETTm1_frwkv_${seq_len}_${pred_len}_seed${seed} \
          --model ${model_name} \
          --data ETTm1 \
          --features M \
          --seq_len ${seq_len} \
          --pred_len ${pred_len} \
          --enc_in 7 \
          --dec_in 7 \
          --c_out 7 \
          --des 'Exp' \
          --embed_size ${embed_size} \
          --d_model ${d_model} \
          --d_ff ${d_model} \
          --batch_size ${batch_size} \
          --learning_rate ${learning_rate} \
          --itr 1 \
          --e_layers 2 \
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
          --train_epochs ${train_epochs} \
          --patience ${patience} \
          --lradj type1 \
          --loss_mode L1 \
          --train_ratio ${train_ratio} \
          --dropout ${dropout} \
          --plot_mat_flag 0 \
          2>&1 | tee -a logs/frwkv/FRWKV_ETTm1_${seq_len}_${pred_len}_seed${seed}.log
        
        # Check if experiment completed successfully
        if [ $? -eq 0 ]; then
            echo "✅ Experiment for prediction length ${pred_len}, random seed ${seed} completed successfully"
        else
            echo "❌ Experiment for prediction length ${pred_len}, random seed ${seed} failed"
        fi
        
        echo "=========================================="
        echo ""
        
        # If not the last experiment, take a break
        if [ $experiment_count -lt $total_experiments ]; then
            echo "Resting for 10 seconds before continuing to next experiment..."
            sleep 10
            echo ""
        fi
    done
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
echo "- Sequence length: ${seq_len}"
echo "- Prediction lengths: ${pred_lengths[@]}"
echo "- Random seeds: ${seeds[@]}"
echo "- Total experiments: ${total_experiments}"
echo ""
echo "Log file list:"
for pred_len in "${pred_lengths[@]}"; do
    for seed in "${seeds[@]}"; do
        log_file="logs/frwkv/FRWKV_ETTm1_${seq_len}_${pred_len}_seed${seed}.log"
        if [ -f "$log_file" ]; then
            echo "✅ ${log_file}"
        else
            echo "❌ ${log_file} (not found)"
        fi
    done
done
echo "=========================================="

# Generate experiment statistics
echo "=========================================="
echo "Experiment statistics"
echo "=========================================="
echo "Prediction length statistics:"
for pred_len in "${pred_lengths[@]}"; do
    count=0
    for seed in "${seeds[@]}"; do
        log_file="logs/frwkv/FRWKV_ETTm1_${seq_len}_${pred_len}_seed${seed}.log"
        if [ -f "$log_file" ]; then
            count=$((count + 1))
        fi
    done
    echo "- Prediction length ${pred_len}: ${count}/${#seeds[@]} experiments completed"
done
echo ""
echo "Random seed statistics:"
for seed in "${seeds[@]}"; do
    count=0
    for pred_len in "${pred_lengths[@]}"; do
        log_file="logs/frwkv/FRWKV_ETTm1_${seq_len}_${pred_len}_seed${seed}.log"
        if [ -f "$log_file" ]; then
            count=$((count + 1))
        fi
    done
    echo "- Random seed ${seed}: ${count}/${#pred_lengths[@]} experiments completed"
done
echo "=========================================="
