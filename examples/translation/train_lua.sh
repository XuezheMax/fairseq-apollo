#! /bin/bash

split=0
seeds=(1 11 65537 101 1999 2017)
seed=${seeds[$split]}

DATA=$1
SAVE_ROOT=$2
model=luna
exp_name=1_apollo_luna_k16_run${split}

SAVE=${SAVE_ROOT}/${exp_name}
rm -rf ${SAVE}
mkdir -p ${SAVE}
src=en
tgt=de
cp $0 ${SAVE}/run.sh

python -u train.py ${DATA} \
    --seed 1 --ddp-backend c10d --find-unused-parameters \
    --valid-subset valid -s $src -t $tgt \
    --eval-bleu --eval-bleu-remove-bpe '@@ ' --best-checkpoint-metric bleu --maximize-best-checkpoint-metric \
    --eval-tokenized-bleu --eval-bleu-detok "space" \
    -a ${model} --encoder-layers 6 --decoder-layers 6 --encoder-projected-length 16 \
    --encoder-projected-attention-heads 1 --decoder-projected-attention-heads 1 \
    --optimizer apollo --lr 10.0 --clip-norm 0.1 --clip-mode 'each' \
    --label-smoothing 0.1 --max-tokens 8192 --share-all-embeddings \
    --dropout 0.1 --attention-dropout 0.1 --activation-dropout 0.1 \
    --lr-scheduler milestone --lr-decay-rate 0.1 --milestones 250000 450000 \
    --weight-decay 1e-8 --weight-decay-type 'decoupled' \
    --criterion label_smoothed_cross_entropy --max-update 500000 \
    --warmup-updates 1000 --warmup-init-lr 0.01 --apollo-beta 0.9 --apollo-eps 1e-4 \
    --keep-last-epochs 10 --keep-interval-updates 1 --update-freq 4 --save-interval-updates 5000 \
    --save-dir ${SAVE} --log-format simple --log-interval 100 --num-workers 0 | tee ${SAVE}/log.txt

# date
# wait

# opt=${SAVE}/test_best.log
# python -u fairseq_cli/generate.py $DATA --gen-subset test -s $src -t $tgt --path ${SAVE}/checkpoint_best.pt --batch-size 300 --remove-bpe "@@ " --beam 5 --max-len-a 2 --max-len-b 0 --quiet | tee ${opt}

# opt=${SAVE}/test_last.log
# python -u fairseq_cli/generate.py $DATA --gen-subset test -s $src -t $tgt --path ${SAVE}/checkpoint_last.pt --batch-size 300 --remove-bpe "@@ " --beam 5 --max-len-a 2 --max-len-b 0 --quiet | tee ${opt}

# python scripts/average_checkpoints.py --inputs ${SAVE} --output ${SAVE}/checkpoint_last10.pt --num-epoch-checkpoints 10
# rm -f ${SAVE}/checkpoint2*.pt
# rm -f ${SAVE}/checkpoint_254_500000.pt

# opt=${SAVE}/test_last10.log
# python -u fairseq_cli/generate.py $DATA --gen-subset test -s $src -t $tgt --path ${SAVE}/checkpoint_last10.pt --batch-size 300 --remove-bpe "@@ " --beam 5 --max-len-a 2 --max-len-b 0 --quiet | tee ${opt}