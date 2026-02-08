SEED=$1

python src/main.py \
--easy True \
--tasks 'Toy' \
--algos 'CRUCB' 'SW_CUCB' 'RUCB' \
--n_random 1 \
--T 500001 \
