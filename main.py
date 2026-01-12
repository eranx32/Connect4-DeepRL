from connect4_env import Connect4Env
import numpy as np
from agent import Agent
from tqdm import tqdm  # אל תשכח להתקין: pip install tqdm
from datetime import datetime
import os

agent = Agent()
env = Connect4Env()
episodes = 450000
start_rival_randomness = 1.0  # היריב מתחיל טיפש לגמרי
end_rival_randomness = 0.05
# פרמטרים לניהול אימון
update_target_every = 1000
global_step_counter = 0
target_update_counter = 0  # רק בשביל התצוגה

print("Starting Training...")

# עוטפים את ה-range ב-tqdm
loop = tqdm(range(1, episodes + 1))


def check_and_update_target():
    # משתמשים ב-global כדי לשנות את המשתנים שמוגדרים בחוץ
    global target_update_counter
    if global_step_counter % update_target_every == 0:
        agent.update_target_network()
        target_update_counter += 1
        # אפשר להוסיף הדפסה קטנה כדי לדעת שזה קרה
        # print(f"Target Network Updated at step {global_step_counter}")


for episode in loop:

    # כאן אנחנו מוודאים שאין הדפסה - show_board=False
    env.reset(show_board=False)
    done = False
    # --- חישוב רמת הקושי של היריב לאפוק הנוכחי ---
    # נוסחה לינארית פשוטה: ככל שמתקדמים במשחקים, הרנדומליות יורדת
    rival_randomness = start_rival_randomness - (episode / episodes) * (start_rival_randomness - end_rival_randomness)
    # מוודאים שלא ירד מתחת למינימום
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

        # --- תור היריב הדינאמי ---
        if np.random.rand() < rival_randomness:
            # היריב עושה טעות / משחק רנדומלי
            rival_action = np.random.choice(env.get_valid_locations())
        else:
            # היריב משחק חכם (היוריסטי)
            rival_action = agent.heuristic_rival(env.board, env.get_valid_locations())

        # היריב משחק. אנחנו שולחים את הלוח המעודכן (שהוא ה-next_state של הסוכן מקודם)
        # שים לב: next_state מהשלב הקודם הופך להיות ה-Current State של היריב
        rival_next_state, rival_reward, rival_done = env.step(rival_action, -1, show_board=False)

        # --- תרחיש ב': היריב ניצח ---
        if rival_done:
            reward = -100  # עונש לסוכן
            done = True
            # שים לב: אנחנו שומרים את המעבר למצב הסופי של הלוח (אחרי תור היריב)
            agent.remember(state, action, reward, rival_next_state, done)
            agent.replay()
            check_and_update_target()  # <--- הנה התיקון: בודקים לפני ששוברים!
            break

        # --- תרחיש ג': המשחק ממשיך ---
        # שומרים את המעבר: מהמצב שהיה לפני התור שלי -> למצב שאחרי התור של היריב
        agent.remember(state, action, reward, rival_next_state, done)
        agent.replay()
        check_and_update_target()  # <--- בדיקה רגילה בסוף כל תור

    if episode % 100 == 0:
        loop.set_postfix({
            'Ep': f"{agent.epsilon:.2f}",
            'RivalRand': f"{rival_randomness:.2f}"
        })

print("\nTraining finished!")

should_save = input("Do you want to save the trained model? (y/n): ")

if should_save.lower() == 'y':
    # יצירת שם קובץ ייחודי עם תאריך ושעה (למשל: connect4_2025-10-27_15-30-00.pth)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    filename = f"connect4_model_{timestamp}.pth"

    # בדיקה אם יש תיקייה למודלים, אם לא - יוצר אותה (כדי שיהיה מסודר)
    if not os.path.exists("models"):
        os.makedirs("models")

    full_path = os.path.join("models", filename)

    agent.save_model(full_path)
    print(f"Model saved successfully to: {full_path}")
else:
    print("Model was NOT saved.")

# --- שלב בדיקה (משחק נגדך) ---
agent.epsilon = 0
while True:
    user_input = input("Ready to play? (y/n): ")
    if user_input.lower() != 'y':
        break

    env.reset(show_board=True)  # במשחק מולך אנחנו כן רוצים לראות לוח!
    done = False
    print("\n--- New Game vs Human ---")

    while not done:
        # תור הסוכן
        print("Agent is thinking...")
        action = agent.act(env.board, env.get_valid_locations())
        print(f"Agent chose column: {action + 1}")
        _, _, done = env.step(action, 1, show_board=True)

        if done:
            print("Agent Won!")
            break

        # תור האדם
        valid_locs = env.get_valid_locations()
        action = agent.manual_act(env.board, valid_locs)
        _, _, done = env.step(action, -1, show_board=True)

        if done:
            print("You Won!")
            break
