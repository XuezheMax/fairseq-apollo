#! /bin/bash

seed=1

DATA=$1
SAVE_ROOT=$2
plen=$3
model=luna_lra_aan

SAVE=${SAVE_ROOT}
mkdir -p ${SAVE}
cp $0 ${SAVE}/run.sh

CUDA_VISIBLE_DEVICES=0,1 python -u train.py ${DATA} \
    --seed $seed --ddp-backend c10d --fp16 --find-unused-parameters \
    -a ${model} --task long_range_arena \
    --optimizer adam --lr 0.0005 --adam-betas '(0.9, 0.98)' --clip-norm 0.0 \
    --encoder-projection-length $plen \
    --best-checkpoint-metric accuracy --maximize-best-checkpoint-metric \
    --encoder-projected-attention-heads 4 \
    --batch-size 16 --sentence-avg --update-freq 1 \
    --lr-scheduler inverse_sqrt --weight-decay 0.0001 \
    --criterion lra_cross_entropy --max-update 20000 --save-interval-updates 2000 \
    --warmup-updates 8000 --warmup-init-lr '1e-07' \
    --keep-last-epochs 10 --keep-interval-updates 10 \
    --save-dir ${SAVE} --log-format simple --log-interval 100 --num-workers 0 \
    --tensorboard-logdir ${SAVE} | tee ${SAVE}/log.txt