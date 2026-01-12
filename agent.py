import numpy as np
import torch
from torch import nn
from connect4Model import Connect4Model
import random
from collections import deque
from connect4Env import check_winner


class Agent:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Agent running on: {self.device}")

        self.gamma = 0.8
        self.epsilon = 1.0
        self.epsilon_min = 0.05
        self.epsilon_decay = 0.999996
        self.learning_rate = 0.0005
        self.batch_size = 256

        self.memory = deque(maxlen=135000)

        self.model = Connect4Model().to(self.device)
        self.target_model = Connect4Model().to(self.device)
        self.update_target_network()
        self.target_model.eval()

        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.learning_rate)
        self.criterion = nn.MSELoss()

    def update_target_network(self):
        self.target_model.load_state_dict(self.model.state_dict())

    def state_to_tensor(self, state):

        player_board = (state == 1).astype(np.float32)
        rival_board = (state == -1).astype(np.float32)

        stacked = np.stack([player_board, rival_board], axis=0)
        return torch.FloatTensor(stacked).unsqueeze(0).to(self.device)

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state, free_columns):

        for col in free_columns:
            temp = state.copy()
            self.simulate_move(temp, col, 1)
            if check_winner(temp, 1):
                return col

        for col in free_columns:
            temp = state.copy()
            self.simulate_move(temp, col, -1)
            if check_winner(temp, -1):
                return col

        if np.random.rand() <= self.epsilon:
            return np.random.choice(free_columns)

        state_tensor = self.state_to_tensor(state)

        with torch.no_grad():
            q_values = self.model(state_tensor).squeeze(0)

        max_q = -float('inf')
        best_col = -1
        for col in free_columns:
            if q_values[col].item() > max_q:
                max_q = q_values[col].item()
                best_col = col

        return best_col if best_col != -1 else np.random.choice(free_columns)

    def heuristic_rival(self, board, free_columns):
        for col in free_columns:
            temp = board.copy()
            self.simulate_move(temp, col, -1)
            if check_winner(temp, -1): return col

        for col in free_columns:
            temp = board.copy()
            self.simulate_move(temp, col, 1)
            if check_winner(temp, 1): return col

        best_score = -10000
        best_col = np.random.choice(free_columns)

        for col in free_columns:
            temp = board.copy()
            self.simulate_move(temp, col, -1)
            score = self.score_position(temp, -1)

            if score > best_score:
                best_score = score
                best_col = col

        return best_col

    def score_position(self, board, piece):
        score = 0

        center_array = [int(i) for i in list(board[:, 3])]
        center_count = center_array.count(piece)
        score += center_count * 3

        for r in range(6):
            row_array = [int(i) for i in list(board[r, :])]
            for c in range(7 - 3):
                window = row_array[c:c + 4]
                score += self.evaluate_window(window, piece)

        for c in range(7):
            col_array = [int(i) for i in list(board[:, c])]
            for r in range(6 - 3):
                window = col_array[r:r + 4]
                score += self.evaluate_window(window, piece)

        for r in range(6 - 3):
            for c in range(7 - 3):
                window = [board[r + i][c + i] for i in range(4)]
                score += self.evaluate_window(window, piece)

        for r in range(6 - 3):
            for c in range(7 - 3):
                window = [board[r + 3 - i][c + i] for i in range(4)]
                score += self.evaluate_window(window, piece)

        return score

    def evaluate_window(self, window, piece):
        score = 0
        opp_piece = -piece

        if window.count(piece) == 4:
            score += 100
        elif window.count(piece) == 3 and window.count(0) == 1:
            score += 5
        elif window.count(piece) == 2 and window.count(0) == 2:
            score += 2

        if window.count(opp_piece) == 3 and window.count(0) == 1:
            score -= 4

        return score

    def simulate_move(self, board, col, player):
        for r in range(5, -1, -1):
            if board[r][col] == 0:
                board[r][col] = player
                break

    def manual_act(self, state, free_columns):
        try:
            a = int(input(f"Col {free_columns + 1}: ")) - 1
            if a in free_columns: return a
        except:
            pass
        return self.manual_act(state, free_columns)

    def replay(self):
        if len(self.memory) < self.batch_size: return

        minibatch = random.sample(self.memory, self.batch_size)

        state_batch = torch.cat([self.state_to_tensor(t[0]) for t in minibatch])
        next_state_batch = torch.cat([self.state_to_tensor(t[3]) for t in minibatch])

        actions = torch.LongTensor([t[1] for t in minibatch]).unsqueeze(1).to(self.device)
        rewards = torch.FloatTensor([t[2] for t in minibatch]).to(self.device)
        dones = torch.FloatTensor([t[4] for t in minibatch]).to(self.device)

        current_q = self.model(state_batch).gather(1, actions).squeeze(1)

        with torch.no_grad():
            max_next_q = self.target_model(next_state_batch).max(1)[0]
            target_q = rewards + (self.gamma * max_next_q * (1 - dones))

        loss = self.criterion(current_q, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def save_model(self, filename):

        torch.save(self.model.state_dict(), filename)