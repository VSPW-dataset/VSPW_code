DATA_DIR='/your/path/to/VSPW'
PRED_DIR='/your/path/to/predictions'

###mIoU

python  evaluator_test.py $DATA_DIR $PRED_DIR

###TC score
python TC_cal.py  $DATA_DIR $PRED_DIR


##VC score

python VC_perclip.py  $DATA_DIR $PRED_DIR
