DATA=$1
SAVE=$2

python -u fairseq_cli/generate.py $DATA --gen-subset test --path ${SAVE}/checkpoint_best.pt --batch-size 300 --remove-bpe "@@ "  --beam 5 --max-len-a 2 --max-len-b 0 --quiet

python -u fairseq_cli/generate.py $DATA --gen-subset test --path ${SAVE}/checkpoint_last.pt --batch-size 300 --remove-bpe "@@ "  --beam 5 --max-len-a 2 --max-len-b 0 --quiet

python scripts/average_checkpoints.py --inputs ${SAVE} --output ${SAVE}/checkpoint_last10.pt --num-epoch-checkpoints 10

python -u fairseq_cli/generate.py $DATA --gen-subset test --path ${SAVE}/checkpoint_last10.pt --batch-size 300 --remove-bpe "@@ " --beam 5 --max-len-a 2 --max-len-b 0 --quiet