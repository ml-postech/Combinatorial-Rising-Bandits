SEED=$1

python src/main.py \
--easy True \
--tasks 'Path' \
--algos 'CRUCB_0.0001' 'CRUCB_0.125' 'CRUCB_0.5' \
--n_random 5 \
--T 300001 \
