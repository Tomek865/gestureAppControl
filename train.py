import os
from stable_baselines3 import PPO
from air_hockey_env import AirHockeyEnv

models_dir = "models/PPO"
log_dir = "logs"

if not os.path.exists(models_dir):
    os.makedirs(models_dir)

if not os.path.exists(log_dir):
    os.makedirs(log_dir)

env = AirHockeyEnv()
env.reset()

model = PPO("MlpPolicy", env, gamma=0.999, verbose=1, tensorboard_log=log_dir)

TIMESTEPS = 10000
iters = 0

print("Rozpoczynam trening... (Naciśnij Ctrl+C aby przerwać)")

while True:
    iters += 1

    model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False)

    model.save(f"{models_dir}/{TIMESTEPS * iters}")

    print(f"Zapisano model: {models_dir}/{TIMESTEPS * iters}.zip")
