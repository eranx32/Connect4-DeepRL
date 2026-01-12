from connect4Env import Connect4Env
import numpy as np
from agent import Agent
from tqdm import tqdm
from datetime import datetime
import os

agent = Agent()
env = Connect4Env()
episodes = 450000
start_rival_randomness = 1.0
end_rival_randomness = 0.05
update_target_every = 1000
global_step_counter = 0
target_update_counter = 0

print("Starting Training...")

loop = tqdm(range(1, episodes + 1))


def check_and_update_target():
    global target_update_counter
    if global_step_counter % update_target_every == 0:
        agent.update_target_network()
        target_update_counter += 1


for episode in loop:

    env.reset(show_board=False)
    done = False
    rival_randomness = start_rival_randomness - (episode / episodes) * (start_rival_randomness - end_rival_randomness)
    rival_randomness = max(rival_randomness, end_rival_randomness)
    while not done:
        global_step_counter += 1

        state = env.board.copy()

        valid_locs = env.get_valid_locations()
        action = agent.act(state, valid_locs)

        next_state, reward, done = env.step(action, 1, show_board=False)

        if done:
            agent.remember(state, action, reward, next_state, done)
            agent.replay()
            check_and_update_target()
            break

        if np.random.rand() < rival_randomness:
            rival_action = np.random.choice(env.get_valid_locations())
        else:
            rival_action = agent.heuristic_rival(env.board, env.get_valid_locations())

        rival_next_state, rival_reward, rival_done = env.step(rival_action, -1, show_board=False)

        if rival_done:
            reward = -100
            done = True
            agent.remember(state, action, reward, rival_next_state, done)
            agent.replay()
            check_and_update_target()
            break

        agent.remember(state, action, reward, rival_next_state, done)
        agent.replay()
        check_and_update_target()

    if episode % 100 == 0:
        loop.set_postfix({
            'Ep': f"{agent.epsilon:.2f}",
            'RivalRand': f"{rival_randomness:.2f}"
        })

print("\nTraining finished!")

should_save = input("Do you want to save the trained model? (y/n): ")

if should_save.lower() == 'y':
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    filename = f"connect4_model_{timestamp}.pth"

    if not os.path.exists("models"):
        os.makedirs("models")

    full_path = os.path.join("models", filename)

    agent.save_model(full_path)
    print(f"Model saved successfully to: {full_path}")
else:
    print("Model was NOT saved.")

agent.epsilon = 0
while True:
    user_input = input("Ready to play? (y/n): ")
    if user_input.lower() != 'y':
        break

    env.reset(show_board=True)
    done = False
    print("\n--- New Game vs Human ---")

    while not done:
        print("Agent is thinking...")
        action = agent.act(env.board, env.get_valid_locations())
        print(f"Agent chose column: {action + 1}")
        _, _, done = env.step(action, 1, show_board=True)

        if done:
            print("Agent Won!")
            break

        valid_locs = env.get_valid_locations()
        action = agent.manual_act(env.board, valid_locs)
        _, _, done = env.step(action, -1, show_board=True)

        if done:
            print("You Won!")
            break
