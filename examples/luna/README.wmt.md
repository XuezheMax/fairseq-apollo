# Traning Luna on Machine Translation

## WMT-14 English-German
Each experiment is conducted on 8 NVIDIA Tesla V100 GPUs. 
To use less GPUs, please increase `--update-freq` accordingly to ensure the same total batch size.

```bash
src=en
tgt=de
MODEL_PATH=/path/to/model

python -u train.py ${DATA} \
    --seed ${seed} --ddp-backend c10d --find-unused-parameters \
    --valid-subset valid -s $src -t $tgt \
    --eval-bleu --eval-bleu-remove-bpe '@@ ' --best-checkpoint-metric ppl \
    --eval-tokenized-bleu --eval-bleu-detok "space" \
    -a [luna_base_tied|luna_base_untied] --encoder-layers 6 --decoder-layers 6 --projection-length 32 \
    --encoder-projected-attention-heads 8 --decoder-projected-attention-heads 8 \
    --optimizer apollo --lr 0.1 --clip-norm 0.1 --clip-mode 'each' \
    --label-smoothing 0.1 --max-tokens 8192 --max-sentences 1024 --share-all-embeddings \
    --dropout 0.1 --attention-dropout 0.1 --activation-dropout 0.1 --word-dropout 0.0 \
    --lr-scheduler milestone --lr-decay-rate 0.1 --milestones 300000 450000 \
    --weight-decay 1e-8 --weight-decay-type 'L2' \
    --criterion label_smoothed_cross_entropy --max-update 500000 \
    --warmup-updates 1000 --warmup-init-lr 0.00001 --apollo-beta 0.9 --apollo-eps 1e-4 \
    --keep-last-epochs 5 --keep-interval-updates 1 --update-freq 1 --save-interval-updates 5000 \
    --save-dir ${MODEL_PATH} --log-format simple --log-interval 100 --num-workers 0 | tee -a ${MODEL_PATH}/log.txt
```
