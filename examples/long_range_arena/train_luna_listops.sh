#! /bin/bash

split=0
seeds=(1 11 65537 101 1999 2017)
seed=${seeds[$split]}

DATA=$1
SAVE_ROOT=$2
plen=$3
model=luna_lra_listop
# exp_name=1_apollo_luna_k16_run${seed}

SAVE=${SAVE_ROOT}
# rm -rf ${SAVE}
mkdir -p ${SAVE}
cp $0 ${SAVE}/run.sh

CUDA_VISIBLE_DEVICES=0,1 python -u train.py ${DATA} \
    --seed $seed --ddp-backend c10d --fp16 --find-unused-parameters \
    -a ${model} --task long_range_arena \
    --optimizer adam --lr 0.0001 --adam-betas '(0.9, 0.98)' --clip-norm 0.0 \
    --encoder-projection-length ${plen} \
    --best-checkpoint-metric accuracy --maximize-best-checkpoint-metric \
    --encoder-projected-attention-heads 8 \
    --apply-bert-init  \
    --batch-size 16 --sentence-avg --update-freq 1 \
    --lr-scheduler inverse_sqrt --weight-decay 0.0001 \
    --criterion lra_cross_entropy --max-update 5000 --save-interval-updates 20 \
    --warmup-updates 1000 --warmup-init-lr '1e-07' \
    --keep-last-epochs 10 --keep-interval-updates 10 \
    --save-dir ${SAVE} --log-format simple --log-interval 100 --num-workers 0 \
    --tensorboard-logdir ${SAVE} | tee ${SAVE}/log.txt