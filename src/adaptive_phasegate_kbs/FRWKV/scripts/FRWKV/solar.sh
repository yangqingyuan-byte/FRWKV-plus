model_name=FRWKV

seq_lens=(96 96 96 96)

pred_lens=(96 192 336 720)
train_ratios=(1.0 1.0 1.0 1.0)

d_models=(512 512 512 512)
embed_size=(16 16 16 16)
cuda_ids1=(0 0 0 0)
dropout=(0.0 0.0 0.0 0.0)
learning_rate=(5e-4 5e-4 5e-4 5e-4)
#epochs=(40 30 40 30)
epochs=(10 10 10 10)

# Random seed
seed=2025

lradj=type1

# Attention mechanism parameters removed, using default values

for ((i = 0; i < 3; i++))
do
  seq_len=${seq_lens[i]}
  pred_len=${pred_lens[i]}
  train_ratio=${train_ratios[i]}
  export CUDA_VISIBLE_DEVICES=${cuda_ids1[i]}

  echo "=========================================="
  echo "Solar experiment: seq_len=${seq_len}, pred_len=${pred_len}"
  echo "=========================================="

  python -u run.py \
    --is_training 1 \
    --root_path ./dataset/Solar/ \
    --data_path solar_AL.txt \
    --model_id Solar_frwkv_${seq_len}_${pred_len}_seed${seed} \
    --model $model_name \
    --data Solar \
    --features M \
    --seq_len ${seq_len} \
    --pred_len ${pred_len} \
    --enc_in 137 \
    --dec_in 137 \
    --c_out 137 \
    --des 'Exp' \
    --embed_size ${embed_size[i]} \
    --d_model ${d_models[i]} \
    --d_ff ${d_models[i]} \
    --batch_size 32 \
    --learning_rate ${learning_rate[i]} \
    --itr 1 \
    --e_layers 2 \
    --lossfun_alpha 0.5 \
    --test_batch_size 16 \
    --test_mode 0 \
    --CKA_flag 0 \
    --fix_seed 1 \
    --seed ${seed} \
    --resume_training 0 \
    --save_every_epoch 0 \
    --use_revin 1 \
    --use_norm 1 \
    --send_mail 0 \
    --save_pdf 1 \
    --train_epochs ${epochs[i]} \
    --patience 5 \
    --lradj ${lradj} \
    --loss_mode L1 \
    --train_ratio $train_ratio \
    --dropout ${dropout[i]} \
    --plot_mat_flag 0 \
    --linear_attention 0 \
    2>&1 | tee -a logs/frwkv/Solar_frwkv_${seq_len}_${pred_len}_seed${seed}.log

  if [ $? -ne 0 ]; then
    echo "❌ Failed: seq_len=${seq_len}, pred_len=${pred_len}"
  else
    echo "✅ Completed: seq_len=${seq_len}, pred_len=${pred_len}"
  fi

  echo "Resting for 5 seconds before continuing..."
  sleep 5
done