# Finetuning RoBERTa on RACE tasks

### 1) Data download and preprocess.
Please refer to RoBERTa repo for more info (https://github.com/pytorch/fairseq/blob/v0.9.0/examples/roberta/README.race.md)

### 3) Fine-tuning on RACE:

```bash
MAX_EPOCH=20           # Number of training epochs.
LR=1e-05              # Peak LR for fixed LR scheduler.
NUM_CLASSES=4
MAX_SENTENCES=8       # Batch size per GPU.
UPDATE_FREQ=2       # Accumulate gradients to simulate training on 8 GPUs.
DATA_DIR=/path/to/data-dir
SAVE_DIR=/path/to/save-dir
LUNA_PATH=/path/to/pretrain/luna

fairseq-train $DATA_DIR \
    --restore-file $LUNA_PATH \
    --reset-optimizer --reset-dataloader --reset-meters \
    --best-checkpoint-metric accuracy --maximize-best-checkpoint-metric \
    --task sentence_ranking \
    --num-classes $NUM_CLASSES \
    --init-token 0 --separator-token 2 \
    --max-option-length 128 \
    --max-positions 512 \
    --shorten-method "truncate" \
    --arch luna_base_untied_512 \
    --warmup-updates 160 \
    --find-unused-parameters \
    --dropout 0.1 --attention-dropout 0.1 --activation-dropout 0. --weight-decay 0.01 \
    --criterion sentence_ranking \
    --optimizer adam --adam-betas '(0.9, 0.98)' --adam-eps 1e-06 \
    --clip-norm 0.0 \
    --save-interval-updates 1000 \
    --keep-last-epochs 1 \
    --lr-scheduler fixed --lr $LR \
    --fp16 --fp16-init-scale 4 --threshold-loss-scale 1 --fp16-scale-window 128 \
    --batch-size $MAX_SENTENCES \
    --required-batch-size-multiple 1 \
    --update-freq $UPDATE_FREQ \
    --max-epoch $MAX_EPOCH \
    --keep-best-checkpoints 5 \
    --keep-interval-updates 10 \
    --save-dir $SAVE_DIR | tee $SAVE_DIR/train.log
```

**Note:**

The above command assumes training on 1 GPU with 24GB of RAM. For GPUs with
less memory, decrease `--max-sentences` and increase `--update-freq`
accordingly to compensate.


### 4) Evaluate
```bash
DATA_DIR=/path/to/data       # data directory used during training
MODEL_PATH=/path/to/save-dir  # path to the finetuned model checkpoint
PREDS_OUT=preds.tsv                     # output file path to save prediction
TEST_SPLIT=test1    # can be test (Middle) or test1 (High)
fairseq-validate \
    $DATA_DIR \
    --valid-subset $TEST_SPLIT \
    --path $MODEL_PATH \
    --batch-size 32 \
    --task sentence_ranking \
    --criterion sentence_ranking \
    --save-predictions $PREDS_OUT
```