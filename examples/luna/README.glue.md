# Finetuning Luna on GLUE tasks

### 1) Download the data from GLUE website (https://gluebenchmark.com/tasks) using following commands:
```bash
wget https://gist.githubusercontent.com/W4ngatang/60c2bdb54d156a41194446737ce03e2e/raw/17b8dd0d724281ed7c3b2aeeda662b92809aadd5/download_glue_data.py
python download_glue_data.py --data_dir glue_data --tasks all
```

### 2) Preprocess GLUE task data:
```bash
./examples/roberta/preprocess_GLUE_tasks.sh glue_data <glue_task_name>
```
`glue_task_name` is one of the following:
`{ALL, QQP, MNLI, QNLI, MRPC, RTE, STS-B, SST-2, CoLA}`
Use `ALL` for preprocessing all the glue tasks.

### 3) Fine-tuning on GLUE task:
Example fine-tuning cmd for `QNLI` task
```bash
TOTAL_NUM_UPDATES=65520  # 20 epochs through QNLI for bsz 32
WARMUP_UPDATES=2000      
LR=1e-05                # Peak LR for polynomial LR scheduler.
NUM_CLASSES=2
MAX_SENTENCES=32        # Batch size.
ACT_DROPOUT=0.1         # Actication dropout
LUNA_PATH=/path/to/luna/model.pt

CUDA_VISIBLE_DEVICES=0 fairseq-train $DATA_PATH \
    --task sentence_prediction --criterion sentence_prediction --num-classes $NUM_CLASSES \
    --best-checkpoint-metric accuracy --maximize-best-checkpoint-metric \
    --reset-optimizer --reset-dataloader --reset-meters \
    --classification-head-name 'cls_classification_head' \
    --arch [luna_base_tied_512|luna_base_untied_512] --restore-file $LUNA_PATH \
    --optimizer adam --lr $LR --adam-eps 1e-06 --adam-betas '(0.9, 0.98)' \
    --lr-scheduler polynomial_decay --total-num-update $TOTAL_NUM_UPDATES --end-learning-rate 0.0 \
    --max-update $MAX_UPDATES --clip-norm 0.0 --weight-decay 0.1 --warmup-updates $WARMUP_UPDATES \
    --dropout 0.1 --activation-dropout ${ACT_DROPOUT} --attention-dropout 0.1 \
    --max-positions 512 --max-sentences $MAX_SENTENCES --max-tokens 8192 \
    --required-batch-size-multiple 1 --init-token 0 --separator-token 2 \
    --log-format simple --save-interval-updates 1000 --find-unused-parameters 
```

For each of the GLUE task, you will need to use following cmd-line arguments:

Model | QNLI | QQP | SST-2 
---|---|---|---
`--num-classes` | 2 | 2 | 2  
`--lr` | 1e-5 | 1e-5 | 1e-5  
`--max-sentences` | 32 | 32 | 32 
`--total-num-update` | 65520 | 228000 | 21050
`--warmup-updates` | 2000 | 6850 | 1263
`--activation-dropout` | 0.1 | 0.1 | 0.0

## GLUE Results

Model | QNLI | QQP | SST-2 
---|---|---|---
`luna.base.tied` | 91.5 | 91.2 | 94.3
`luna.base.untied` | 92.2 | 91.3 | 94.6
