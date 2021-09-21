#!/bin/bash
ENTITIES="EpiCustomV3 Large_DatasetV3"
MAX_LENGTH=192

for ENTITY in $ENTITIES
do
	echo "***** " $ENTITY " Preprocessing Start *****"
	DATA_DIR=./datasets/$ENTITY

	# Replace tab to space
	cat $DATA_DIR/train.tsv | tr '\t' ' ' > $DATA_DIR/train.txt.tmp
	cat $DATA_DIR/val.tsv | tr '\t' ' ' > $DATA_DIR/val.txt.tmp
	cat $DATA_DIR/test.tsv | tr '\t' ' ' > $DATA_DIR/test.txt.tmp
	echo "Replacing Done"

	# Preprocess for BERT-based models
	python3 preprocess.py $DATA_DIR/train.txt.tmp dmis-lab/biobert-large-cased-v1.1 $MAX_LENGTH > $DATA_DIR/train.txt
	python3 preprocess.py $DATA_DIR/val.txt.tmp dmis-lab/biobert-large-cased-v1.1 $MAX_LENGTH > $DATA_DIR/val.txt
	python3 preprocess.py $DATA_DIR/test.txt.tmp dmis-lab/biobert-large-cased-v1.1 $MAX_LENGTH > $DATA_DIR/test.txt
	cat $DATA_DIR/train.txt $DATA_DIR/val.txt $DATA_DIR/test.txt | cut -d " " -f 2 | grep -v "^$"| sort | uniq > $DATA_DIR/labels.txt
	echo "***** " $ENTITY " Preprocessing Done *****"
done
