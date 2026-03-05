import os
from stable_baselines3 import PPO
from air_hockey_env import AirHockeyEnv

models_dir = "models\PPO"
log_dir = "logs"

env = AirHockeyEnv()
env.reset()
model_path = f"{models_dir}/100000000.zip"

model = PPO.load(model_path, env=env, tensorboard_log=log_dir)

print(f"Wznowiono trening z pliku: {model_path}")
print("Kontynuuje nauke...")

TIMESTEPS = 10000
iters = 100

while True:
    iters += 1

    model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False)

    model.save(f"{models_dir}/{TIMESTEPS * iters}")
    print(f"Zapisano model: {models_dir}/{TIMESTEPS * iters}.zip")
