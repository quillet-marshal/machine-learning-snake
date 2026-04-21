import snake_func as snakeGame
import random
import time
import calendar
import math
import statistics
from copy import deepcopy
from collections import namedtuple, deque
from itertools import count

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


dateTime = time.gmtime()
timestamp_start = calendar.timegm(dateTime)
dateTimeFormatted = str(dateTime[0]) + "/" + str(dateTime[1]) + "/" + str(dateTime[2]) + " " + str(dateTime[3]) + ":" + str(dateTime[4]) + ":" + str(dateTime[5])

device = torch.device(
    "cuda" if torch.cuda.is_available() else
    "mps" if torch.backends.mps.is_available() else
    "cpu"
)

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))


class ReplayMemory(object):
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

class DQN(nn.Module):

    def __init__(self, n_observations, n_actions):
        super(DQN, self).__init__()
        self.layer1 = nn.Linear(n_observations, 128)
        self.layer2 = nn.Linear(128, 128)
        self.layer3 = nn.Linear(128, n_actions)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)


BATCH_SIZE = 128 # BATCH_SIZE is the number of transitions sampled from the replay buffer
GAMMA = 0.99 # GAMMA is the discount factor as mentioned in the previous section
EPS_START = 0.9 # EPS_START is the starting value of epsilon
EPS_END = 0.01 # EPS_END is the final value of epsilon
EPS_DECAY = 2500 # EPS_DECAY controls the rate of exponential decay of epsilon, higher means a slower decay
TAU = 0.005 # TAU is the update rate of the target network
LR = 3e-4 # LR is the learning rate of the ``AdamW`` optimizer


n_actions = 4 # up, left, down, right
n_observations = 52 # number of state observations

## Uncomment the line below and update the path to load a saved model
# savedModel = "saved_model.pth"

policy_net = DQN(n_observations, n_actions).to(device)
try:
    print("Trying to loading saved model from path:", savedModel)
    policy_net.load_state_dict(torch.load(savedModel, weights_only=True))
    # policy_net.load_state_dict(torch.load(savedModel, weights_only=True), strict=False)
except:
    new_model = True
    print("No valid saved model provided. A new model will be trained.")
else:
    new_model = False
    print("Successfully loaded saved model.")

    policy_net.eval()
    # policy_net.train()

target_net = DQN(n_observations, n_actions).to(device)
target_net.load_state_dict(policy_net.state_dict())


optimizer = optim.AdamW(policy_net.parameters(), lr=LR, amsgrad=True)
memory = ReplayMemory(10000)

steps_done = 0

def select_action(state):
    global steps_done
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * \
        math.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1
    if sample > eps_threshold:
        with torch.no_grad():
            # t.max(1) will return the largest column value of each row.
            # second column on max result is index of where max element was
            # found, so we pick action with the larger expected reward.
            return policy_net(state).max(1).indices.view(1, 1)
    else:
        randomAction = random.randrange(0, 4)
        return torch.tensor([[randomAction]], device=device, dtype=torch.long)

def optimize_model():
    if len(memory) < BATCH_SIZE:
        return
    transitions = memory.sample(BATCH_SIZE)
    # Transpose the batch, converts batch-array of Transitions to Transition of batch-arrays.
    batch = Transition(*zip(*transitions))

    # Compute mask of non-final states and concatenate batch elements (final state is one after which simulation ended)
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                          batch.next_state)), device=device, dtype=torch.bool)
    non_final_next_states = torch.cat([s for s in batch.next_state
                                                if s is not None])
    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
    # columns of actions taken. These are the actions which would've been taken
    # for each batch state according to policy_net
    state_action_values = policy_net(state_batch).gather(1, action_batch)

    # Compute V(s_{t+1}) for all next states.
    # Expected values of actions for non_final_next_states are computed based
    # on the "older" target_net; selecting their best reward with max(1).values
    # This is merged based on the mask, such that we'll have either the expected
    # state value or 0 in case the state was final.
    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    with torch.no_grad():
        next_state_values[non_final_mask] = target_net(non_final_next_states).max(1).values
    # Compute the expected Q values
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    # Compute Huber loss
    criterion = nn.SmoothL1Loss()
    loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    # In-place gradient clipping
    torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100)
    optimizer.step()


if torch.cuda.is_available() or torch.backends.mps.is_available():
    num_episodes = 5000
else:
    num_episodes = 50


tilesAcross = 5 # number of gameboard rows/columns
snakeDict = None
apple = None
action = None
greatestLength = 0
timeOutTimer = 10 * tilesAcross
timeOut = timeOutTimer

finalScores = []
timeOutCounter = 0
offscreenDeathCounter = 0
selfcollisionDeathCounter = 0

for i_episode in range(num_episodes):
    # Initialize the environment and get its state
    state = snakeGame.play(tilesWide=tilesAcross, n_observations=n_observations)
    state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
    for t in count():
        action = select_action(state)
        observation, reward, terminated, snakeDict, apple, offScreen, selfCollision = snakeGame.play(snakeDict=snakeDict, apple=apple, action=action, tilesWide=tilesAcross, n_observations=n_observations)
        if reward >= 100:
            timeOut += 10 * tilesAcross
        reward = torch.tensor([reward], device=device)
        done = terminated

        if terminated or timeOut < 1:
            lastLength = len(snakeDict)
            finalScores.append(lastLength)
            if lastLength > greatestLength:
                greatestLength = lastLength
                bestModel = deepcopy(policy_net.state_dict())
            
            next_state = None
            snakeDict = None
            apple = None
        else:
            next_state = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)

        # Store the transition in memory
        memory.push(state, action, next_state, reward)

        # Move to the next state
        state = next_state

        # Perform one step of the optimization (on the policy network)
        optimize_model()

        # Soft update of the target network's weights
        # θ′ ← τ θ + (1 −τ )θ′
        target_net_state_dict = target_net.state_dict()
        policy_net_state_dict = policy_net.state_dict()
        for key in policy_net_state_dict:
            target_net_state_dict[key] = policy_net_state_dict[key]*TAU + target_net_state_dict[key]*(1-TAU)
        target_net.load_state_dict(target_net_state_dict)

        if done:
            if offScreen:
                offscreenDeathCounter += 1
            elif selfCollision:
                selfcollisionDeathCounter += 1

            print("Episode", i_episode + 1, "of", num_episodes, "done. Final length was", lastLength)
            timeOut = timeOutTimer
            break
        elif timeOut < 1:
            print("Episode", i_episode + 1, "of", num_episodes, "timed out. Final length was", lastLength)
            timeOut = timeOutTimer
            timeOutCounter += 1
            break
        else:
            timeOut -= 1


timestamp_end = calendar.timegm(time.gmtime())
policy_filepath = "saved_training\pytorchSnake_policy_file_" + str(tilesAcross) + "_tilesWide_" + str(n_observations) + "_observations_" + str(num_episodes) + "_rounds_" + str(greatestLength) + "_maxLen_" + str(timestamp_end) + ".pth"
torch.save(bestModel, policy_filepath)

runningTimeSeconds = timestamp_end - timestamp_start
runningTimeMinutes = runningTimeSeconds / 60
averageLength = statistics.mean(finalScores)
print("Complete. Greatest length of snake was", greatestLength, "and the average length was", averageLength, "in a total of", num_episodes, "episodes and total time elapsed was approx.", str(runningTimeMinutes), "minutes.")

recordData = "\r" + dateTimeFormatted + "," + str(n_observations) + "," + str(tilesAcross) + "," + str(num_episodes) + "," + str(new_model) + "," + str(greatestLength) + "," + str(averageLength) + "," + str(timeOutCounter) + "," + str(offscreenDeathCounter) + "," + str(selfcollisionDeathCounter) + "," + str(runningTimeSeconds)
with open("Analysis\\records.csv", "a") as record:
    record.write(recordData)
