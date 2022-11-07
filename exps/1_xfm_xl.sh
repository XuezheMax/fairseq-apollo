#! /bin/bash
#SBATCH --output=slurm_logs/slurm-%A-%a.out
#SBATCH --error=slurm_logs/slurm-%A-%a.err
#SBATCH --partition=devlab
#SBATCH --job-name=xfm.xl
#SBATCH --comment="iclr rebuttal"
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=10
#SBATCH --mem=100GB
#SBATCH --signal=USR1@90
#SBATCH --open-mode=append
#SBATCH --time=3-00:00:00
#SBATCH --array=0
##SBATCH --wckey=submitit

# command
module load cuda/11.3
source activate mega

split=$SLURM_ARRAY_TASK_ID
split=1
seeds=(22 42 65537 8191 131071)
seed=${seeds[$split]}

export WANDB_PROJECT=mega_lm
export WANDB_TEAM=mega_lm
export WANDB_WATCH="false"

LR=5e-3
CHUNK=2048
bsz=8

WARMUP=24000
WEIGHT_DECAY=0.1
TOTAL_NUM_UPDATES=400000

DATE=`date +%Y%m%d`
SAVE_ROOT=saved_models/xfm_xl
DATA=/data/home/chuntinz/checkpoint/research/fairseq-apollo/data-bin/enwik8
model=transformer_xl_lm_base
exp_name=1_pg19_1024_xl_lr${LR}_warmup${WARMUP}_seed${seed}

SAVE=${SAVE_ROOT}/${exp_name}
rm -rf ${SAVE}
mkdir -p ${SAVE}
cp $0 ${SAVE}/run.sh

#export MASTER_ADDR=${SLURM_NODELIST:0:20}${SLURM_NODELIST:21:3}
#export MASTER_PORT=15127
#export WORLD_SIZE=8

#srun --label 
python -u train.py ${DATA} \
    --wandb-project ${WANDB_PROJECT} --wandb-entity ${WANDB_TEAM} \
    --seed ${seed} --ddp-backend no_c10d --max-target-positions 406 \
    --valid-subset valid --task language_modeling -a ${model} \
    --normalize-before --tgt_len ${chunk_size} --mem_len ${chunk_size} --is-book --is-xfm-xl \
    --batch-size ${bsz} --update-freq 1 \
    --optimizer adam --lr ${LR} --adam-betas '(0.9, 0.98)' --adam-eps 1e-8 --clip-norm 1.0 \
    --lr-scheduler linear_decay --total-num-update ${TOTAL_NUM_UPDATES} --end-learning-rate 0.0 \
    --warmup-updates ${WARMUP} --warmup-init-lr '1e-07' \
    --criterion 'cross_entropy' --share-decoder-input-output-embed \
    --dropout 0.1 --attention-dropout 0.0 --weight-decay ${WEIGHT_DECAY} \
    --max-update ${TOTAL_NUM_UPDATES} \
    --no-epoch-checkpoints \
    --sample-break-mode 'none' --fp16 \
    --save-dir ${SAVE} --log-format simple --log-interval 100 --num-workers 0 | tee -a ${SAVE}/log.txt


python -u train.py ${DATA} \
    --restore-file "checkpoint_best.pt" --eval-only --same_length --clamp_len 1000 \
    --mem_len 2048 \
    --seed ${seed} --ddp-backend no_c10d --max-target-positions 406 \
    --valid-subset valid --task language_modeling -a ${model} \
    --normalize-before --tgt_len ${chunk_size} --mem_len ${chunk_size} --is-book --is-xfm-xl \
    --batch-size ${bsz} --update-freq 1 \
    --optimizer adam --lr ${LR} --adam-betas '(0.9, 0.98)' --adam-eps 1e-8 --clip-norm 1.0 \
    --lr-scheduler linear_decay --total-num-update ${TOTAL_NUM_UPDATES} --end-learning-rate 0.0 \
    --warmup-updates ${WARMUP} --warmup-init-lr '1e-07' \
    --criterion 'cross_entropy' --weight-decay ${WEIGHT_DECAY} \
    --max-update ${TOTAL_NUM_UPDATES} \
    --no-epoch-checkpoints \
    --sample-break-mode 'none' --fp16 \
    --save-dir ${SAVE} --log-format simple --log-interval 100 --num-workers 0 | tee -a ${SAVE}/log.txt
