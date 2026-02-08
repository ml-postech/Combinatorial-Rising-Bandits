SEED=$1

python src/main.py \
--easy True \
--tasks 'Spanning' \
--algos 'CRUCB' 'SW_CUCB' 'SW_CTS' 'RUCB' 'SW_UCB' 'SW_TS' \
--n_random 5 \
--T 300001 \
