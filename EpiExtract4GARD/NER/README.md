# Named Entity Recognition
This code was adapted from the [Pytorch implementation of BioBERT](https://github.com/dmis-lab/biobert-pytorch) created by DMIS Lab. Every file except *preprocess.py* was modified. 

Before training, please run `./preprocess.sh` to preprocess the datasets downloaded in `biobert-pytorch` (see [here](https://github.com/jhyuklee/biobert-pytorch)).

To train an NER model with BioBERT-v1.1 (base), run the command below.

## Sample Training Commands
```bash
export SAVE_DIR=./outputLG5
export DATA_DIR=../datasets
export MAX_LENGTH=192
export SAVE_STEPS=1500
export SEED=1
export ENTITY=Large_DatasetV2
export NUM_EPOCHS=30
export LEARNING_RATE=5e-5
export WEIGHT_DECAY=0.01
export BATCH_SIZE=16
export EPSILON=1e-8

python3 run_ner.py \
    --data_dir ${DATA_DIR}/${ENTITY}/ \
    --labels ${DATA_DIR}/${ENTITY}/labels.txt \
    --model_name_or_path dmis-lab/biobert-base-cased-v1.1 \
    --learning_rate ${LEARNING_RATE} \
    --output_dir ${SAVE_DIR} \
    --weight_decay ${WEIGHT_DECAY} \
    --adam_epsilon ${EPSILON} \
    --max_seq_length ${MAX_LENGTH} \
    --num_train_epochs ${NUM_EPOCHS} \
    --per_device_train_batch_size ${BATCH_SIZE} \
    --save_steps ${SAVE_STEPS} \
    --seed ${SEED} \
    --do_train \
    --do_eval \
    --do_predict \
    --overwrite_output_dir
```

## V2 Evaluation Results

| Test #      | Dataset  | epochs | learning rate | weight decay | batch size | adam epsilon | loss     | precision | recall   | F1       |
| ----------- | :------: | ------ | ------------- | ------------ | ---------- | ------------ | -------- | --------- | -------- | -------- |
| output      | CustomV2 | 1      | 0.00005       | 0            | 32         | 1E-08        | 0.037207 | 0.785075  | 0.781575 | 0.783321 |
| output1     | CustomV2 | 2      | 0.00005       | 0            | 16         | 1E-08        | 0.026727 | 0.869955  | 0.864785 | 0.867362 |
| output2     | CustomV2 | 2      | 0.00005       | 0            | 32         | 1E-08        | 0.028645 | 0.834795  | 0.84844  | 0.841562 |
| output3     | CustomV2 | 2      | 0.00005       | 0.01         | 16         | 1E-08        | 0.026742 | 0.871257  | 0.864785 | 0.868009 |
| output4     | CustomV2 | 2      | 0.00005       | 0.01         | 32         | 1E-08        | 0.028791 | 0.828446  | 0.839525 | 0.833948 |
| output5     | CustomV2 | 2      | 0.00005       | 0            | 16         | 1E-08        | 0.039709 | 0.753247  | 0.775632 | 0.764275 |
| output6     | CustomV2 | 2      | 0.00005       | 0            | 16         | 0.00001      | 0.050409 | 0.63369   | 0.704309 | 0.667136 |
| output7     | CustomV2 | 2      | 0.00005       | 0.01         | 16         | 1E-08        | 0.039709 | 0.753247  | 0.775632 | 0.764275 |
| output8     | CustomV2 | 2      | 0.00005       | 0.02         | 16         | 0.00001      | 0.050473 | 0.630667  | 0.702823 | 0.664793 |
| output9     | CustomV2 | 3      | 0.00005       | 0            | 16         | 1E-08        | 0.028544 | 0.849112  | 0.852897 | 0.851001 |
| output10    | CustomV2 | 3      | 0.00005       | 0.05         | 16         | 1E-08        | 0.02848  | 0.858434  | 0.846954 | 0.852655 |
| output11    | CustomV2 | 3      | 0.00003       | 0            | 16         | 1E-08        | 0.029121 | 0.826979  | 0.838039 | 0.832472 |
| output12    | CustomV2 | 3      | 0.00003       | 0.1          | 16         | 1E-08        | 0.029219 | 0.826725  | 0.836553 | 0.83161  |
| output13    | LargeV2  | 2      | 0.00005       | 0.01         | 16         | 1E-08        | 0.02706  | 0.850812  | 0.855869 | 0.853333 |
| output14    | LargeV2  | 2      | 0.00005       | 0.01         | 16         | 1E-10        | 0.027364 | 0.855224  | 0.851412 | 0.853313 |
| output15    | LargeV2  | 2      | 0.00005       | 0.01         | 10         | 1E-08        | 0.028575 | 0.837681  | 0.858841 | 0.848129 |
| output16    | LargeV2  | 2      | 0.00005       | 0.01         | 10         | 1E-10        | 0.028765 | 0.828326  | 0.860327 | 0.844023 |
| output17    | LargeV2  | 2      | 0.00007       | 0.01         | 16         | 1E-08        | 0.027057 | 0.830657  | 0.845468 | 0.837997 |
| output18    | LargeV2  | 2      | 0.00007       | 0.01         | 16         | 1E-10        | 0.026644 | 0.846954  | 0.846954 | 0.846954 |
| output19    | LargeV2  | 2      | 0.00007       | 0.01         | 10         | 1E-08        | 0.028384 | 0.844118  | 0.852897 | 0.848485 |
| output20    | LargeV2  | 2      | 0.00007       | 0.01         | 10         | 1E-10        | 0.028517 | 0.843658  | 0.849926 | 0.84678  |
| outputLG1   | LargeV2  | 2      | 0.00005       | 0.01         | 16         | 1E-08        | 0.02706  | 0.850812  | 0.855869 | 0.853333 |
| outputCSTM1 | CustomV2 | 2      | 0.00005       | 0.01         | 16         | 1E-08        | 0.026742 | 0.871257  | 0.864785 | 0.868009 |
| outputLG2   | LargeV2  | 3      | 0.00005       | 0.01         | 16         | 1E-08        | 0.028842 | 0.848397  | 0.864785 | 0.856512 |
| outputCSTM2 | CustomV2 | 3      | 0.00005       | 0.01         | 16         | 1E-08        | 0.028983 | 0.853293  | 0.846954 | 0.850112 |
| outputLG3   | LargeV2  | 4      | 0.00005       | 0.01         | 16         | 1E-08        | 0.033489 | 0.859031  | 0.869242 | 0.864106 |
| outputCSTM3 | CustomV2 | 4      | 0.00005       | 0.01         | 16         | 1E-08        | 0.031214 | 0.87145   | 0.86627  | 0.868852 |
| outputLG4   | LargeV2  | 20     | 0.00005       | 0.01         | 16         | 1E-08        | 0.054767 | 0.839311  | 0.869242 | 0.854015 |
| outputCSTM4 | CustomV2 | 20     | 0.00005       | 0.01         | 16         | 1E-08        | 0.053693 | 0.888554  | 0.876672 | 0.882573 |
| outputLG5   | LargeV2  | 30     | 0.00005       | 0.01         | 16         | 1E-08        | 0.055439 | 0.883756  | 0.881129 | 0.88244  |
| outputCSTM5 | CustomV2 | 30     | 0.00005       | 0.01         | 16         | 1E-08        | 0.054413 | 0.891239  | 0.876672 | 0.883895 |

## Contact
For help or issues using BioBERT-PyTorch, please create an issue and tag [@minstar](https://github.com/minstar).
