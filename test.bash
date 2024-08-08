python3 general_main.py --data HRS --num_tasks 6 --cl_type nc --agent SRTFD --num_runs 1 --N 1000
python3 general_main.py --data HRS --num_tasks 6 --cl_type vc --agent SRTFD --num_runs 1  --N 1000 

python3 general_main.py --data TEP --num_tasks 22 --cl_type nc --agent SRTFD --num_runs 1 --N 1000
python3 general_main.py --data TEP --num_tasks 22 --cl_type vc --agent SRTFD --num_runs 1 --N 1000

python3 general_main.py --data CARLS_S --num_tasks 10 --cl_type nc --agent SRTFD --num_runs 3 --N 1000
python3 general_main.py --data CARLS_S --num_tasks 10 --cl_type vc --agent SRTFD --num_runs 1 --N 1000  

python3 general_main.py --data CARLS_M --num_tasks 5 --cl_type nc --agent SRTFD --num_runs 3 --N 1000
python3 general_main.py --data CARLS_M --num_tasks 5 --cl_type vc --agent SRTFD --num_runs 1 --N 1000  


# ER
python3 general_main.py --data HRS --num_tasks 6 --cl_type nc --agent ER --num_runs 1 --N 1000
python3 general_main.py --data HRS --num_tasks 6 --cl_type vc --agent ER --num_runs 1 --N 1000

# ASER
python3 general_main.py --data HRS --num_tasks 6 --cl_type nc --agent ER --num_runs 1 --N 1000 --update ASER --retrieve ASER --mem_size 5000 --aser_type asvm --n_smp_cls 1.5 --k 3 
python3 general_main.py --data HRS --num_tasks 6 --cl_type vc --agent ER --num_runs 1 --ns_factor 0.4 --update ASER --retrieve ASER --mem_size 5000 --aser_type asvm --n_smp_cls 1.5 --k 3  --n_r 1 --f_r 1

# AGEM
python3 general_main.py --data HRS --num_tasks 6 --cl_type nc --agent AGEM --num_runs 1 --N 1000 --retrieve random --update random --mem_size 5000
python3 general_main.py --data HRS --num_tasks 6 --cl_type vc --agent AGEM --num_runs 1 --ns_factor 0.4 --retrieve random --update random --mem_size 5000  --n_r 1 --f_r 1

# MPOS_RVFL
python3 general_main.py --data HRS --num_tasks 6 --cl_type nc --agent MPOS_RVFL --num_runs 1 --N 1000 --n_anchor 50
python3 general_main.py --data HRS --num_tasks 6 --cl_type vc --agent MPOS_RVFL --num_runs 1 --ns_factor 0.4 --N 1000 --n_anchor 50  --n_r 1 --f_r 1
