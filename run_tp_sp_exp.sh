#!/bin/bash

# 可配置参数
OUT_DIR="tmp"
EXP_NAME="dp"
TP=1
CP=1
PP=1
DP=4
MODEL_NAME="HuggingFaceTB/SmolLM-135M-Instruct"
NUM_ATTENTION_HEADS=16
NUM_KEY_VALUE_HEADS=4
GRAD_ACC_STEPS=1
MBS=16
SEQ_LEN=128

# 第一部分：生成配置
python create_config.py \
    --out_dir ${OUT_DIR} \
    --exp_name ${EXP_NAME} \
    --tp ${TP} \
    --cp ${CP} \
    --pp ${PP} \
    --dp ${DP} \
    --model_name ${MODEL_NAME} \
    --num_attention_heads ${NUM_ATTENTION_HEADS} \
    --num_key_value_heads ${NUM_KEY_VALUE_HEADS} \
    --grad_acc_steps ${GRAD_ACC_STEPS} \
    --mbs ${MBS} \
    --seq_len ${SEQ_LEN}

# 第二部分：启动训练
CUDA_DEVICE_MAX_CONNECTIONS=1 torchrun \
    --nproc_per_node ${DP} \
    --master_addr localhost \
    --master_port 25500 \
    ddp_exp.py \
    --config tmp/dp/config.json


# 张量并行可配置参数
OUT_DIR="tmp"
EXP_NAME="tp"
TP=4
CP=1
PP=1
DP=1
MODEL_NAME="HuggingFaceTB/SmolLM-135M-Instruct"
NUM_ATTENTION_HEADS=16
NUM_KEY_VALUE_HEADS=4
GRAD_ACC_STEPS=1
MBS=16
SEQ_LEN=128

# 第一部分：生成配置
python create_config.py \
    --out_dir ${OUT_DIR} \
    --exp_name ${EXP_NAME} \
    --tp ${TP} \
    --cp ${CP} \
    --pp ${PP} \
    --dp ${DP} \
    --model_name ${MODEL_NAME} \
    --num_attention_heads ${NUM_ATTENTION_HEADS} \
    --num_key_value_heads ${NUM_KEY_VALUE_HEADS} \
    --grad_acc_steps ${GRAD_ACC_STEPS} \
    --mbs ${MBS} \
    --seq_len ${SEQ_LEN}

# 第二部分：启动训练
CUDA_DEVICE_MAX_CONNECTIONS=1 torchrun \
    --nproc_per_node ${TP} \
    --master_addr localhost \
    --master_port 25500 \
    tp_exp.py \
    --config tmp/tp/config.json

CUDA_DEVICE_MAX_CONNECTIONS=1 torchrun \
    --nproc_per_node ${TP} \
    --master_addr localhost \
    --master_port 25500 \
    tp_sp_exp.py \
    --config tmp/tp/config.json