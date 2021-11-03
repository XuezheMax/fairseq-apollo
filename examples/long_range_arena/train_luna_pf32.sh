#! /bin/bash

seed=1

DATA=$1
SAVE_ROOT=$2
plen=$3
model=luna_lra_pf32

SAVE=${SAVE_ROOT}
# rm -rf ${SAVE}
mkdir -p ${SAVE}
cp $0 ${SAVE}/run.sh

CUDA_VISIBLE_DEVICES=0,1 python -u train.py ${DATA} \
    --seed $seed --ddp-backend c10d --fp16 \
    -a ${model} --task long_range_arena \
    --encoder-projection-length $plen --find-unused-parameters \
    --encoder-projected-attention-heads 4 --act-dropout 0. --dropout 0.2 \
    --best-checkpoint-metric accuracy --maximize-best-checkpoint-metric \
    --optimizer adam --lr 0.001 --adam-betas '(0.9, 0.98)' --clip-norm 0.0 \
    --batch-size 256 --sentence-avg --update-freq 1 \
    --lr-scheduler inverse_sqrt --weight-decay 0.0 \
    --criterion lra_cross_entropy --max-update 62500 \
    --warmup-updates 4000 --warmup-init-lr '1e-07' \
    --keep-last-epochs 10 \
    --save-dir ${SAVE} --log-format simple --log-interval 100 --num-workers 0 \
    --tensorboard-logdir ${SAVE} | tee ${SAVE}/log.txt