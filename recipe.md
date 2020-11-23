For all experiments, the recommended random seeds are in {1, 11, 101, 65537, 524287}

# SGD
```base
python -u train.py <DATA PATH> \
    --seed <RANDOM SEED> \
    -s en -t de --valid-subset valid \
    --eval-bleu --eval-bleu-remove-bpe '@@ ' --best-checkpoint-metric bleu --maximize-best-checkpoint-metric \
    --eval-tokenized-bleu --eval-bleu-detok "space" \
    -a 'transformer_wmt_en_de' --optimizer sgd --lr 0.1 --momentum 0.9 --clip-norm 1.0 \
    --label-smoothing 0.1 --max-tokens 8192 --share-all-embeddings \
    --attention-dropout 0.1 --activation-dropout 0.1 \
    --lr-scheduler milestone --lr-decay-rate 0.1 --milestones 250000 450000 --weight-decay 1e-6 \
    --criterion label_smoothed_cross_entropy --max-update 500000 \
    --warmup-updates 1000 --warmup-init-lr 0.0001 --dropout 0.1 \
    --save-dir <MODEL PATH> \
    --keep-last-epochs 10 --keep-interval-updates 1 --update-freq 1 --save-interval-updates 5000 \
    --log-format simple --log-interval 100 --num-workers 0
```

# Adam
```base
python -u train.py <DATA PATH> \
    --seed <RANDOM SEED> \
    -s en -t de --valid-subset valid \
    --eval-bleu --eval-bleu-remove-bpe '@@ ' --best-checkpoint-metric bleu --maximize-best-checkpoint-metric \
    --eval-tokenized-bleu --eval-bleu-detok "space" \
    -a 'transformer_wmt_en_de' --optimizer adam --lr 0.0005 --adam-betas '(0.9, 0.98)' --clip-norm 1.0 \
    --label-smoothing 0.1 --max-tokens 8192 --share-all-embeddings \
    --attention-dropout 0.1 --activation-dropout 0.1 \
    --lr-scheduler inverse_sqrt --weight-decay 0.0001 \
    --criterion label_smoothed_cross_entropy --max-update 500000 \
    --warmup-updates 4000 --warmup-init-lr '1e-07' --dropout 0.1 \
    --save-dir <MODEL PATH> \
    --keep-last-epochs 10 --keep-interval-updates 1 --update-freq 1 --save-interval-updates 5000 \
    --log-format simple --log-interval 100 --num-workers 0
```

# RAdam
```base
python -u train.py <DATA PATH> \
    --seed <RANDOM SEED> \
    -s en -t de --valid-subset valid \
    --eval-bleu --eval-bleu-remove-bpe '@@ ' --best-checkpoint-metric bleu --maximize-best-checkpoint-metric \
    --eval-tokenized-bleu --eval-bleu-detok "space" \
    -a 'transformer_wmt_en_de' --optimizer radam --lr 0.0005 --radam-betas '(0.9, 0.999)' --clip-norm 1.0 \
    --label-smoothing 0.1 --max-tokens 8192 --share-all-embeddings \
    --attention-dropout 0.1 --activation-dropout 0.1 \
    --lr-scheduler milestone --lr-decay-rate 0.1 --milestones 250000 450000 --weight-decay 0.0001 \
    --criterion label_smoothed_cross_entropy --max-update 500000 \
    --warmup-updates 0 --warmup-init-lr -1 --dropout 0.1 \
    --save-dir <MODEL PATH> \
    --keep-last-epochs 10 --keep-interval-updates 1 --update-freq 1 --save-interval-updates 5000 \
    --log-format simple --log-interval 100 --num-workers 0
```

# Apollo
```base
python -u train.py <DATA PATH> \
    --seed <RANDOM SEED> \
    -s en -t de --valid-subset valid \
    --eval-bleu --eval-bleu-remove-bpe '@@ ' --best-checkpoint-metric bleu --maximize-best-checkpoint-metric \
    --eval-tokenized-bleu --eval-bleu-detok "space" \
    -a 'transformer_wmt_en_de' --optimizer apollo --lr 10 --apollo-beta 0.9 --apollo-eps 1e-4 --clip-norm 1.0 \
    --label-smoothing 0.1 --max-tokens 8192 --share-all-embeddings \
    --attention-dropout 0.1 --activation-dropout 0.1 \
    --lr-scheduler milestone --lr-decay-rate 0.1 --milestones 250000 450000 --weight-decay 1e-8 \
    --criterion label_smoothed_cross_entropy --max-update 500000 \
    --warmup-updates 1000 --warmup-init-lr 0.01 --dropout 0.1 \
    --save-dir <MODEL PATH> \
    --keep-last-epochs 10 --keep-interval-updates 1 --update-freq 1 --save-interval-updates 5000 \
    --log-format simple --log-interval 100 --num-workers 0 
```
