SEED=$1

python src/main.py \
--easy True \
--tasks 'Max' \
--algos 'CRUCB_0.01' 'CRUCB_0.05' 'CRUCB_0.125' 'CRUCB_0.25' 'CRUCB_0.4' 'CRUCB_0.5' \
--n_random 5 \
--T 300001 \
