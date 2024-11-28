# DDPG and PPO comparison for the Gymnasium MoutainCarEnv-v0 and Pendulum-v1

## Set up 


Create a python virtual environment, source it, install torch separadly to avoid cuda packages if you don't have a GPU, then the other required packages. It might take a while.
```sh
python -m venv venv
source venv/bin/activate # (Linux)
venv\Scripts\activate # (Windowns)
pip install torch --index-url https://download.pytorch.org/whl/cpu 
pip install -r requirements.txt
```

Compare DDPG and PPO in the same number of timesteps and computational time: