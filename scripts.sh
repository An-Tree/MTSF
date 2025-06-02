export CUDA_VISIBLE_DEVICES=0
model_name=MTSF
python -u run.py \
  --is_training 1 \
  --root_path ./dataset/ \
  --data_path ETTh1.csv \
  --model_id ETTh1_96_96 \
  --model $model_name \
  --data ETTh1 \
  --dec_in 7 \
  --seq_len 96 \
  --pred_len 96 \
  --m_layers 2 \
  --d_layers 3 \
  --n_heads 16 \
  --lambda_ 0.8 \
  --beta 0.1 \
  --d_model 256 \
  --d_ff 256 \
  --batch_size 256 \
  --m 3 \
  --r 1e-1 \
  --train_epochs 20 \
  --patch_len 24 \
  --stride 24 \
  --xp_i
