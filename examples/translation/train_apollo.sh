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
    -a ${arch} --optimizer apollo --lr 10 --apollo-beta 0.9 --apollo-eps 1e-4 \
    --label-smoothing 0.1 --max-tokens 8192 --share-all-embeddings \
    --attention-dropout 0.1 --activation-dropout 0.1 \
    --lr-scheduler milestone --lr-decay-rate 0.1 --milestones 150000 250000 \
    --weight-decay 1e-8 --weight-decay-type 'L2' \
    --criterion label_smoothed_cross_entropy --max-update 300000 \
    --encoder-attention-heads 2 --decoder-attention-heads 2 \
    --warmup-updates 1000 --warmup-init-lr 0.01 --dropout 0.1 \
    --save-dir ${model_dir} --clip-norm 0.1 --clip-mode 'each' \
    --keep-last-epochs 10 --keep-interval-updates 1 --update-freq 4 --save-interval-updates 5000 \
    --log-format simple --log-interval 100 --num-workers 0 --tensorboard-logdir ${model_dir} \
    | tee ${model_dir}/train.log
