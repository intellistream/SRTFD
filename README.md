# SRTFD: Scalable Real-time Fault Diagnosis

SRTFD is a continual-learning system for real-time fault diagnosis over
evolving industrial data streams. This repository is the DataSys research
artifact accompanying the
[ICDE 2025 paper](https://doi.org/10.1109/ICDE65448.2025.00328).

## Project status

The public TEP and CARLS experiment paths are included. The HRS dataset is
private and cannot be used for public reproduction, so results depending on it
require separate authorized access.

## Reproducibility

**1. Environment Requirements**

Ensure you have all the necessary Python packages by installing them from the provided `requirements.txt` file.

**2. Data Sources**

- **HRS Dataset**: This dataset is private and specific to the requirements of cooperative factories.

- **TEP and CARLS Datasets**: These two datasets are included in our publicly available code.

**3. Setting Up and Running SRTFD**

i. **Install Required Packages**

   Make sure you have Python installed. Then, navigate to the project directory and install the required packages using the following command:

   ```bash
   pip install -r requirements.txt
   ```

ii. **Run SRTFD**

   Execute the main script to start the SRTFD process:

   ```bash
   python3 general_main.py --data TEP --num_tasks 22 --cl_type nc --agent SRTFD --num_runs 1 --N 1000

   python3 general_main.py --data TEP --num_tasks 22 --cl_type vc --agent SRTFD --num_runs 1 --N 1000

   python3 general_main.py --data CARLS_S --num_tasks 10 --cl_type nc --agent SRTFD --num_runs 1 --N 1000

   python3 general_main.py --data CARLS_S --num_tasks 10 --cl_type vc --agent SRTFD --num_runs 1 --N 1000  

   python3 general_main.py --data CARLS_M --num_tasks 5 --cl_type nc --agent SRTFD --num_runs 1 --N 1000

   python3 general_main.py --data CARLS_M --num_tasks 5 --cl_type vc --agent SRTFD --num_runs 1 --N 1000
   ```

**Additional Resources**

For more detailed instructions and documentation, please refer to the project's `test.bash` file or the official documentation provided with the project.

## License

The repository is licensed under Apache License 2.0. Reused continual-learning
components and datasets retain their upstream notices and terms.
