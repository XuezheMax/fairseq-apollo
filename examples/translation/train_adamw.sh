databin_dir=$1
model_dir=$2
arch=$3
mkdir -p ${model_dir}
cp $0 ${model_dir}/train.sh

python -u train.py ${databin_dir} \
    --seed 1 \
    -s en -t de --valid-subset valid \
    --eval-bleu --eval-bleu-remove-bpe '@@ ' --best-checkpoint-metric bleu --maximize-best-checkpoint-metric \
    --eval-tokenized-bleu --eval-bleu-detok "space" \
    -a $arch --optimizer adam --lr 0.0005 --adam-betas '(0.9, 0.98)' --clip-norm 1.0 \
    --label-smoothing 0.1 --max-tokens 8192 --share-all-embeddings \
    --attention-dropout 0.1 --activation-dropout 0.1 \
    --lr-scheduler inverse_sqrt --weight-decay 0.0001 \
    --criterion label_smoothed_cross_entropy --max-update 300000 \
    --warmup-updates 4000 --warmup-init-lr '1e-07' --dropout 0.1 \
    --save-dir ${model_dir}  \
    --keep-last-epochs 10 --keep-interval-updates 1 --update-freq 4 --save-interval-updates 5000 \
    --log-format simple --log-interval 100 --num-workers 0 --tensorboard-logdir ${model_dir} \
    | tee ${model_dir}/train.log
