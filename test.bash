python3 general_main.py --data HRS --num_tasks 6 --cl_type nc --agent SRTFD --num_runs 1 --N 1000
python3 general_main.py --data HRS --num_tasks 6 --cl_type vc --agent SRTFD --num_runs 1 --ns_factor 0.4 --N 1000 --n_r 1 --f_r 1

python3 general_main.py --data TEP --num_tasks 22 --cl_type nc --agent SRTFD --num_runs 1 --N 1000
python3 general_main.py --data TEP --num_tasks 22 --cl_type vc --agent SRTFD --num_runs 1 --ns_factor 0.4  --n_r 1 --f_r 1

python3 general_main.py --data CARLS_S --num_tasks 10 --cl_type nc --agent SRTFD --num_runs 3 --N 1000
python3 general_main.py --data CARLS_S --num_tasks 10 --cl_type vc --agent SRTFD --num_runs 1 --N 1000  --n_r 1 --f_r 1

python3 general_main.py --data CARLS_M --num_tasks 5 --cl_type nc --agent SRTFD --num_runs 3 --N 1000
python3 general_main.py --data CARLS_M --num_tasks 5 --cl_type vc --agent SRTFD --num_runs 1 --N 1000  --n_r 1 --f_r 1


# ER
python3 general_main.py --data HRS --num_tasks 6 --cl_type nc --agent ER --num_runs 1 --N 1000
python3 general_main.py --data HRS --num_tasks 6 --cl_type vc --agent ER --num_runs 1 --ns_factor 0.4  --n_r 1 --f_r 1

# SCR
python3 general_main.py --data HRS --num_tasks 6 --cl_type nc --agent SCR --num_runs 1 --N 1000 --retrieve random --update random --mem_size 5000 --head mlp --temp 0.07 --eps_mem_batch 100
python3 general_main.py --data HRS --num_tasks 6 --cl_type vc --agent SCR --num_runs 1 --ns_factor 0.4 --retrieve random --update random --mem_size 5000 --head mlp --temp 0.07 --eps_mem_batch 100  --n_r 1 --f_r 1

# ASER
python3 general_main.py --data HRS --num_tasks 6 --cl_type nc --agent ER --num_runs 1 --N 1000 --update ASER --retrieve ASER --mem_size 5000 --aser_type asvm --n_smp_cls 1.5 --k 3 
python3 general_main.py --data HRS --num_tasks 6 --cl_type vc --agent ER --num_runs 1 --ns_factor 0.4 --update ASER --retrieve ASER --mem_size 5000 --aser_type asvm --n_smp_cls 1.5 --k 3  --n_r 1 --f_r 1

# AGEM
python3 general_main.py --data HRS --num_tasks 6 --cl_type nc --agent AGEM --num_runs 1 --N 1000 --retrieve random --update random --mem_size 5000
python3 general_main.py --data HRS --num_tasks 6 --cl_type vc --agent AGEM --num_runs 1 --ns_factor 0.4 --retrieve random --update random --mem_size 5000  --n_r 1 --f_r 1

# CNDPM
python3 general_main.py --data HRS --num_tasks 6 --cl_type nc --agent CNDPM --num_runs 1 --N 1000 --stm_capacity 1000 --classifier_chill 0.01 --log_alpha -300
python3 general_main.py --data HRS --num_tasks 6 --cl_type vc --agent CNDPM --num_runs 1 --ns_factor 0.4 --stm_capacity 1000 --classifier_chill 0.01 --log_alpha -300  --n_r 1 --f_r 1

# MPOS_RVFL
python3 general_main.py --data HRS --num_tasks 6 --cl_type nc --agent MPOS_RVFL --num_runs 1 --N 1000 --n_anchor 50
python3 general_main.py --data HRS --num_tasks 6 --cl_type vc --agent MPOS_RVFL --num_runs 1 --ns_factor 0.4 --N 1000 --n_anchor 50  --n_r 1 --f_r 1