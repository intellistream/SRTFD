# SRTFD: Scalable Real-time Fault Diagnosis

```markdown
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
```
