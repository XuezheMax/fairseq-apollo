# Finetuning Luna on Commonsense QA
### 1) Data download and preprocess.
Please refer to RoBERTa repo for more info (https://github.com/pytorch/fairseq/blob/v0.9.0/examples/roberta/commonsense_qa/README.md)

### 2) Finetune

```bash
MAX_UPDATES=8000      # Number of training steps.
WARMUP_UPDATES=150    # Linearly increase LR over this many steps.
LR=1e-05              # Peak LR for polynomial LR scheduler.
MAX_SENTENCES=32      # Batch size.
SEED=1              # Random seed.
LUNA_PATH=/path/to/pretrained/luna
DATA_DIR=/path/to/data-dir
SAVE_DIR=/path/to/save-dir
mkdir -p $SAVE_DIR
touch ${SAVE_DIR}/train.log

# we use the --user-dir option to load the task from
# the examples/roberta/commonsense_qa directory:
FAIRSEQ_PATH=/usr1/home/xiangk/fairseq-apollo-bert/fairseq-apollo
FAIRSEQ_USER_DIR=${FAIRSEQ_PATH}/examples/roberta/commonsense_qa

fairseq-train --fp16  \
    $DATA_DIR \
    --user-dir $FAIRSEQ_USER_DIR \
    --restore-file $LUNA_PATH \
    --reset-optimizer --reset-dataloader --reset-meters \
    --no-epoch-checkpoints --no-last-checkpoints --no-save-optimizer-state \
    --best-checkpoint-metric accuracy --maximize-best-checkpoint-metric \
    --task commonsense_qa --init-token 0 --bpe gpt2 \
    --arch luna_base_untied_512 --max-positions 512 \
    --dropout 0.2 --attention-dropout 0.0 --activation-dropout 0.1 --weight-decay 0.01 \
    --criterion sentence_ranking --num-classes 5 \
    --optimizer adam --adam-betas '(0.9, 0.98)' --adam-eps 1e-06 --clip-norm 0.0 \
    --lr-scheduler polynomial_decay --lr $LR \
    --warmup-updates $WARMUP_UPDATES --total-num-update $MAX_UPDATES \
    --batch-size $MAX_SENTENCES \
    --max-update $MAX_UPDATES \
    --save-interval-updates 100 \
    --keep-best-checkpoints 1 \
    --keep-interval-updates 1 \
    --log-format simple --log-interval 25 \
    --seed $SEED --save-dir ${SAVE_DIR} | tee ${SAVE_DIR}/train.log
```

The above command assumes training on 1 GPU with 24GB of RAM. For GPUs with
less memory, decrease `--max-sentences` and increase `--update-freq`
accordingly to compensate.

### 3) Evaluate
```python
import json
import torch
from fairseq.models.luna_bert import LunaBertModel
from examples.luna_bert import commonsense_qa  # load the Commonsense QA task
# Change these paths accordingly.
luna = LunaBertModel.from_pretrained('save-dir', 'checkpoint_best.pt', 'csqa-data')
luna.eval()  # disable dropout
luna.cuda()  # use the GPU (optional)
nsamples, ncorrect = 0, 0
with open('../../LRA-data/csqa//valid.jsonl') as h:
    for line in h:
        example = json.loads(line)
        scores = []
        for choice in example['question']['choices']:
            input = luna.encode(
                'Q: ' + example['question']['stem'],
                'A: ' + choice['text'],
                no_separator=True
            )
            score = luna.predict('sentence_classification_head', input, return_logits=True)
            scores.append(score)
        pred = torch.cat(scores).argmax()
        answer = ord(example['answerKey']) - ord('A')
        nsamples += 1
        if pred == answer:
            ncorrect += 1

print('Accuracy: ' + str(ncorrect / float(nsamples)))
```