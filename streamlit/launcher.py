import subprocess

cmd = "python -m streamlit run /home/cdsw/CB2MUHX/project-weeks-promptu/streamlit/pages/prompt_optimization.py --server.port 8100 --server.address 127.0.0.1"

subprocess.run(cmd, shell=True)