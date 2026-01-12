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
        self.epsilon_min = 0.05  # העליתי טיפה את המינימום, שיישאר קצת ערני
        self.epsilon_decay = 0.999996  # האטה קלה בדעיכה
        self.learning_rate = 0.0005
        self.batch_size = 256  # הגדלתי Batch - עוזר ליציבות

        self.memory = deque(maxlen=135000)  # הגדלתי זיכרון

        self.model = Connect4Model().to(self.device)
        self.target_model = Connect4Model().to(self.device)
        self.update_target_network()
        self.target_model.eval()

        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.learning_rate)
        self.criterion = nn.MSELoss()

    def update_target_network(self):
        self.target_model.load_state_dict(self.model.state_dict())

    def state_to_tensor(self, state):
        """
        פונקציית קסם: הופכת לוח פשוט (6,7)
        לטנסור מפוצל (2,6,7) שמפריד בין השחקן ליריב.
        """
        # ערוץ 0: הכלים של השחקן הנוכחי (1 בלוח)
        player_board = (state == 1).astype(np.float32)
        # ערוץ 1: הכלים של היריב (-1 בלוח)
        rival_board = (state == -1).astype(np.float32)

        # איחוד לערימה אחת
        stacked = np.stack([player_board, rival_board], axis=0)
        return torch.FloatTensor(stacked).unsqueeze(0).to(self.device)  # (1, 2, 6, 7)

    def remember(self, state, action, reward, next_state, done):
        # שומרים את ה-State הגולמי (numpy), את ההמרה נעשה ב-Replay כדי לחסוך זיכרון RAM
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state, free_columns):
        # --- שלב 1: רשת ביטחון - בדיקת ניצחון מיידי (שח-מט) ---
        # לפני הכל, בודקים אם יש מהלך שמסיים את המשחק לטובתי
        for col in free_columns:
            temp = state.copy()
            self.simulate_move(temp, col, 1)  # מניח שהסוכן הוא תמיד 1
            if check_winner(temp, 1):
                # print(f"Immediate Win found at col {col}") # אפשר להוריד את ההערה לבדיקה
                return col

        # --- שלב 2: רשת ביטחון - מניעת הפסד מיידי ---
        # אם אין ניצחון, בודקים אם חייבים לחסום את היריב עכשיו
        for col in free_columns:
            temp = state.copy()
            self.simulate_move(temp, col, -1)  # מניח שהיריב הוא -1
            if check_winner(temp, -1):
                # print(f"Blocking immediate threat at col {col}")
                return col

        if np.random.rand() <= self.epsilon:
            return np.random.choice(free_columns)

        # שימוש בפונקציית ההמרה החדשה
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

    # --- הבוס הגדול: יריב היוריסטי (מורה) ---
    def heuristic_rival(self, board, free_columns):
        # 1. בדיקת ניצחון מיידי (שלו)
        for col in free_columns:
            temp = board.copy()
            self.simulate_move(temp, col, -1)
            if check_winner(temp, -1): return col

        # 2. בדיקת חסימה מיידית (שלך)
        for col in free_columns:
            temp = board.copy()
            self.simulate_move(temp, col, 1)
            if check_winner(temp, 1): return col

        # 3. בחירה חכמה מבוססת ניקוד (כולל אלכסונים!)
        best_score = -10000
        best_col = np.random.choice(free_columns)

        for col in free_columns:
            temp = board.copy()
            self.simulate_move(temp, col, -1)
            score = self.score_position(temp, -1)  # שולחים את הלוח ואת מי שאנחנו רוצים לקדם (-1)

            if score > best_score:
                best_score = score
                best_col = col

        return best_col

    def score_position(self, board, piece):
        score = 0

        # בונוס על מרכז הלוח (אסטרטגיה בסיסית)
        center_array = [int(i) for i in list(board[:, 3])]
        center_count = center_array.count(piece)
        score += center_count * 3

        # סריקה אופקית
        for r in range(6):
            row_array = [int(i) for i in list(board[r, :])]
            for c in range(7 - 3):
                window = row_array[c:c + 4]
                score += self.evaluate_window(window, piece)

        # סריקה אנכית
        for c in range(7):
            col_array = [int(i) for i in list(board[:, c])]
            for r in range(6 - 3):
                window = col_array[r:r + 4]
                score += self.evaluate_window(window, piece)

        # סריקה אלכסונית (חיובית)
        for r in range(6 - 3):
            for c in range(7 - 3):
                window = [board[r + i][c + i] for i in range(4)]
                score += self.evaluate_window(window, piece)

        # סריקה אלכסונית (שלילית)
        for r in range(6 - 3):
            for c in range(7 - 3):
                window = [board[r + 3 - i][c + i] for i in range(4)]
                score += self.evaluate_window(window, piece)

        return score

    def evaluate_window(self, window, piece):
        score = 0
        opp_piece = -piece  # אם אני 1, היריב -1, ולהפך

        if window.count(piece) == 4:
            score += 100
        elif window.count(piece) == 3 and window.count(0) == 1:
            score += 5  # מעודד יצירת שלישיות
        elif window.count(piece) == 2 and window.count(0) == 2:
            score += 2  # מעודד יצירת זוגות

        # אסטרטגיה הגנתית - אם ליריב יש 3, תוריד לי ניקוד (זה מצב מסוכן)
        if window.count(opp_piece) == 3 and window.count(0) == 1:
            score -= 4

        return score

    def simulate_move(self, board, col, player):
        for r in range(5, -1, -1):
            if board[r][col] == 0:
                board[r][col] = player
                break

    def manual_act(self, state, free_columns):
        # המרה לטנסור בשביל המודל (אם נרצה להציג ניבויים בעתיד)
        # כרגע רק קולט מהמשתמש
        try:
            a = int(input(f"Col {free_columns + 1}: ")) - 1
            if a in free_columns: return a
        except:
            pass
        return self.manual_act(state, free_columns)

    def replay(self):
        if len(self.memory) < self.batch_size: return

        minibatch = random.sample(self.memory, self.batch_size)

        # הכנת Batchים
        # שים לב: אנחנו צריכים להמיר כל לוח בנפרד לטנסור כפול
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
        """
        שומר את משקולות המודל (state_dict) לקובץ.
        """
        torch.save(self.model.state_dict(), filename)