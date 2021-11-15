from collections import deque
import os
import random
from tqdm import tqdm

import torch

from utils_drl import Agent
from utils_env import MyEnv
from utils_memory import ReplayMemory

# Discount Rate
GAMMA = 0.99
GLOBAL_SEED = 0
MEM_SIZE = 100_000
RENDER = False
SAVE_PREFIX = "./models"
# The number of Input Frame
STACK_SIZE = 4

# Epsilon policy
EPS_START = 1.
EPS_END = 0.1
EPS_DECAY = 1000000

# Sample Size when learn
BATCH_SIZE = 32
# The number of turns that learn
POLICY_UPDATE = 4
# sync with target network and policy network
TARGET_UPDATE = 10_000
# The number of turns that not learning
WARM_STEPS = 50_000
# The number of turns that training
MAX_STEPS = 50_000_000
# The gap between testings
EVALUATE_FREQ = 100_000

rand = random.Random()
rand.seed(GLOBAL_SEED)
new_seed = lambda: rand.randint(0, 1000_000)

torch.manual_seed(new_seed())
# decide running in GPU or CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Create a breakout env
env = MyEnv(device)
# Create a player (can load from a trained model)
agent = Agent(
    env.get_action_dim(),
    device,
    GAMMA,
    new_seed(),
    EPS_START,
    EPS_END,
    EPS_DECAY,
    # "./models/sss"
)
# Memory replay
memory = ReplayMemory(STACK_SIZE + 1, MEM_SIZE, device)

#### Training ####
# Use for sampling screen
obs_queue: deque = deque(maxlen=5)
done = True
# visualize the training process
progressive = tqdm(range(MAX_STEPS), total=MAX_STEPS,
                   ncols=50, leave=False, unit="b")

for step in progressive:
    if done:
        # A game finished. Start a new game
        observations, _, _ = env.reset()
        for obs in observations:
            obs_queue.append(obs)

    # training is True when running for WARM_STEPS turns
    training = len(memory) > WARM_STEPS
    # packing to a stuck structure for agent
    state = env.make_state(obs_queue).to(device).float()
    # get an action through agent network
    action = agent.run(state, training)
    # get the next state, reward, from envirement
    obs, reward, done = env.step(action)
    # add new state in the obs_queue(for making stuck structure for agent)
    obs_queue.append(obs)
    # add in the memory
    memory.push(env.make_folded_state(obs_queue), action, reward, done)

    # Time to LEARN!
    if step % POLICY_UPDATE == 0 and training:
        agent.learn(memory, BATCH_SIZE)

    # Time to SYNC!
    if step % TARGET_UPDATE == 0:
        agent.sync()

    # Time to make a testing!
    if step % EVALUATE_FREQ == 0:
        avg_reward, frames = env.evaluate(obs_queue, agent, render=RENDER)
        with open("rewards.txt", "a") as fp:
            fp.write(f"{step//EVALUATE_FREQ:3d} {step:8d} {avg_reward:.1f}\n")
        if RENDER:
            prefix = f"eval_{step//EVALUATE_FREQ:03d}"
            os.mkdir(prefix)
            for ind, frame in enumerate(frames):
                with open(os.path.join(prefix, f"{ind:06d}.png"), "wb") as fp:
                    frame.save(fp, format="png")
        agent.save(os.path.join(
            SAVE_PREFIX, f"model_{step//EVALUATE_FREQ:03d}"))
        done = True
