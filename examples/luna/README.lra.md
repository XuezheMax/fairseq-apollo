# Training Luna on LRA tasks

### 1) Download the data from LRA repo (https://github.com/google-research/long-range-arena)

### 2) Preprocess data:
```bash
fairseq-preprocess \
        --only-source \
        --trainpref ${TRAIN_INPUT} \
        --validpref ${VALID_INPUT} \
        --testpref ${TEST_INPUT} \
        --destdir "$OUT_FOLDER/src-bin" \
        --workers 10
fairseq-preprocess \
        --only-source \
        --trainpref ${TRAIN_LABEL} \
        --validpref ${VALID_LABEL} \
        --testpref ${TEST_LABEL} \
        --destdir "$OUT_FOLDER/label-bin" \
        --workers 10
```

### 3) Train on a LRA task:
```bash
# Set up training envs. Same for all tasks.
seed=1

DATA=/path/to/data-dir
SAVE_ROOT=/path/to/save-dir
plen=16 # Projection length for Luna, could be 128, 256.
SAVE=${SAVE_ROOT}
mkdir -p ${SAVE}
cp $0 ${SAVE}/run.sh
```
```bash
# listops
model=luna_lra_listop lra
python -u train.py ${DATA} \
    --seed $seed --ddp-backend c10d --fp16 --find-unused-parameters \
    -a ${model} --task long_range_arena \
    --optimizer adam --lr 0.0001 --adam-betas '(0.9, 0.98)' --clip-norm 0.0 \
    --encoder-projection-length ${plen} \
    --best-checkpoint-metric accuracy --maximize-best-checkpoint-metric \
    --apply-bert-init  --encoder-layers 1 \
    --batch-size 32 --sentence-avg --update-freq 1 \
    --lr-scheduler inverse_sqrt --weight-decay 0.0001 \
    --criterion lra_cross_entropy --max-update 5000 --save-interval-updates 20 \
    --warmup-updates 1000 --warmup-init-lr '1e-07' \
    --keep-last-epochs 10 --keep-interval-updates 10 \
    --save-dir ${SAVE} --log-format simple --log-interval 100 --num-workers 0 \
    --tensorboard-logdir ${SAVE} | tee ${SAVE}/log.txt
```
```bash
# text
model=luna_lra_imdb
python -u train.py ${DATA} \
    --seed $seed --ddp-backend c10d --fp16 --find-unused-parameters \
    -a ${model} --task long_range_arena \
    --optimizer adam --lr 0.00005  \
    --dropout 0.3 --attention-dropout 0.3 \
    --adam-betas '(0.9, 0.98)' --clip-norm 0.0 \
    --encoder-projection-length $plen \
    --best-checkpoint-metric accuracy --maximize-best-checkpoint-metric \
    --batch-size 32 --sentence-avg --update-freq 1 \
    --lr-scheduler inverse_sqrt --weight-decay 0.0001 \
    --criterion lra_cross_entropy --max-update 20000 --save-interval-updates 100 \
    --warmup-updates 8000 --warmup-init-lr '1e-07' \
    --keep-last-epochs 10 --keep-interval-updates 10 \
    --save-dir ${SAVE} --log-format simple --log-interval 100 --num-workers 0 \
    --tensorboard-logdir ${SAVE} | tee ${SAVE}/log.txt
```
```bash
# retrieval
model=luna_lra_aan
python -u train.py ${DATA} \
    --seed $seed --ddp-backend c10d --fp16 --find-unused-parameters \
    -a ${model} --task long_range_arena \
    --optimizer adam --lr 0.0005 --adam-betas '(0.9, 0.98)' --clip-norm 0.0 \
    --encoder-projection-length $plen \
    --best-checkpoint-metric accuracy --maximize-best-checkpoint-metric \
    --batch-size 32 --sentence-avg --update-freq 1 \
    --lr-scheduler inverse_sqrt --weight-decay 0.0001 \
    --criterion lra_cross_entropy --max-update 20000 --save-interval-updates 2000 \
    --warmup-updates 8000 --warmup-init-lr '1e-07' \
    --keep-last-epochs 10 --keep-interval-updates 10 \
    --save-dir ${SAVE} --log-format simple --log-interval 100 --num-workers 0 \
    --tensorboard-logdir ${SAVE} | tee ${SAVE}/log.txt
```
```bash
# cifar10
model=luna_lra_cifar10
python -u train.py ${DATA} \
    --seed $seed --ddp-backend c10d --fp16 \
    -a ${model} --task long_range_arena \
    --encoder-projection-length $plen --find-unused-parameters \
    --attention-dropout 0.1 \
    --best-checkpoint-metric accuracy --maximize-best-checkpoint-metric \
    --optimizer adam --lr 0.005 --dropout 0.3 --adam-betas '(0.9, 0.98)' --clip-norm 0.0 \
    --batch-size 256 --sentence-avg --update-freq 1 \
    --lr-scheduler inverse_sqrt --weight-decay 0.0 \
    --criterion lra_cross_entropy --max-update 35000 \
    --warmup-updates 4000 --warmup-init-lr '1e-07' \
    --keep-last-epochs 10 \
    --save-dir ${SAVE} --log-format simple --log-interval 100 --num-workers 0 \
    --tensorboard-logdir ${SAVE} | tee ${SAVE}/log.txt
```
```bash
# pf32
model=luna_lra_pf32
python -u train.py ${DATA} \
    --seed $seed --ddp-backend c10d --fp16 \
    -a ${model} --task long_range_arena \
    --encoder-projection-length $plen --find-unused-parameters \
    --dropout 0.2 \
    --best-checkpoint-metric accuracy --maximize-best-checkpoint-metric \
    --optimizer adam --lr 0.001 --adam-betas '(0.9, 0.98)' --clip-norm 0.0 \
    --batch-size 512 --sentence-avg --update-freq 1 \
    --lr-scheduler inverse_sqrt --weight-decay 0.0 \
    --criterion lra_cross_entropy --max-update 62500 \
    --warmup-updates 4000 --warmup-init-lr '1e-07' \
    --keep-last-epochs 10 \
    --save-dir ${SAVE} --log-format simple --log-interval 100 --num-workers 0 \
    --tensorboard-logdir ${SAVE} | tee ${SAVE}/log.txt
```
**Note:**

Above cmd-args and hyperparams are tested on one Nvidia `TITAN RTX` GPU with `24gb` of memory for each task. Depending on the GPU memory resources available to you, you can use increase `--update-freq`.
