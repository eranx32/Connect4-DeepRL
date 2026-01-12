import pygame
import sys
import math
import numpy as np
import torch
import os
import glob
from connect4Env import Connect4Env, check_winner
from agent import Agent

# --- הגדרות ---
SQUARESIZE = 100
RADIUS = int(SQUARESIZE / 2 - 5)
ROW_COUNT = 6
COLUMN_COUNT = 7
width = COLUMN_COUNT * SQUARESIZE
height = (ROW_COUNT + 1) * SQUARESIZE
size = (width, height)

# צבעים לגיבוי (אם אין תמונות)
BLUE = (0, 0, 255)
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)


class AssetManager:
    def __init__(self):
        self.use_images = False
        self.board_img = None
        self.red_img = None
        self.yellow_img = None

        # נסיון לטעון תמונות
        try:
            if os.path.exists('assets/board.png'):
                loaded_board = pygame.image.load('assets/board.png')
                # אם יש תמונה, חייבים לוודא שהיא תומכת בשקיפות (convert_alpha)
                self.board_img = pygame.transform.smoothscale(loaded_board,
                                                              (width, height - SQUARESIZE)).convert_alpha()

                loaded_red = pygame.image.load('assets/red.png')
                self.red_img = pygame.transform.smoothscale(loaded_red, (SQUARESIZE, SQUARESIZE)).convert_alpha()

                loaded_yellow = pygame.image.load('assets/yellow.png')
                self.yellow_img = pygame.transform.smoothscale(loaded_yellow, (SQUARESIZE, SQUARESIZE)).convert_alpha()

                self.use_images = True
                print("Assets loaded successfully!")
        except Exception as e:
            print(f"Could not load images: {e}. Using procedural graphics.")
            self.use_images = False

        # --- התיקון הגדול: יצירת לוח פרוצדורלי עם שקיפות ---
        if not self.use_images:
            # אנחנו יוצרים משטח חדש שיהיה הלוח שלנו
            self.board_surf = pygame.Surface((width, height - SQUARESIZE))

            # צובעים את כולו בכחול
            self.board_surf.fill(BLUE)

            # מציירים עליו את העיגולים בשחור
            for c in range(COLUMN_COUNT):
                for r in range(ROW_COUNT):
                    center = (int(c * SQUARESIZE + SQUARESIZE / 2), int(r * SQUARESIZE + SQUARESIZE / 2))
                    pygame.draw.circle(self.board_surf, BLACK, center, RADIUS)

            # מילת הקסם! אומרים לפייתון שכל מה ששחור במשטח הזה - הופך לשקוף
            self.board_surf.set_colorkey(BLACK)

    def draw_board_overlay(self, screen):
        if self.use_images:
            screen.blit(self.board_img, (0, SQUARESIZE))
        else:
            # עכשיו אנחנו פשוט מדביקים את המשטח שהכנו מראש (עם החורים השקופים)
            screen.blit(self.board_surf, (0, SQUARESIZE))

    def draw_piece(self, screen, col, row, player, y_offset=0):
        # ... (שאר הפונקציה נשאר אותו דבר כמו בתיקון הקודם) ...

        # חישוב המיקום
        if y_offset != 0:
            y = y_offset
            x = col * SQUARESIZE
        else:
            y = (row + 1) * SQUARESIZE
            x = col * SQUARESIZE

        if self.use_images:
            img = self.red_img if player == 1 else self.yellow_img
            screen.blit(img, (x, y))
        else:
            center = (int(x + SQUARESIZE / 2), int(y + SQUARESIZE / 2))
            color = (200, 0, 0) if player == 1 else (200, 200, 0)
            highlight = (255, 50, 50) if player == 1 else (255, 255, 50)

            pygame.draw.circle(screen, color, center, RADIUS)
            pygame.draw.circle(screen, highlight, (center[0] - 5, center[1] - 5), RADIUS - 10)


def get_latest_model_path():
    list_of_files = glob.glob('models/*.pth')
    if not list_of_files: return None
    return max(list_of_files, key=os.path.getctime)


def animate_drop(screen, assets, board_state, col, row, player):
    """
    פונקציה חוסמת שמבצעת את האנימציה של הנפילה והקפיצה
    """
    # פיזיקה
    y = 0  # מתחילים מלמעלה
    target_y = (row + 1) * SQUARESIZE
    velocity = 0
    gravity = 2.5  # כוח משיכה
    bounciness = 0.4  # כמה חזק הקפיצה חזרה (0 עד 1)

    clock = pygame.time.Clock()
    animating = True

    while animating:
        # 1. חישוב פיזיקלי
        velocity += gravity
        y += velocity

        # בדיקת פגיעה בקרקע
        if y >= target_y:
            y = target_y
            velocity = -velocity * bounciness  # היפוך מהירות (קפיצה)

            # אם המהירות נמוכה מדי, מפסיקים לקפוץ
            if abs(velocity) < 5:
                animating = False

        # 2. ציור הפריים
        screen.fill(BLACK)  # ניקוי מסך

        # א. ציור כל הדיסקיות הסטטיות שכבר קיימות בלוח
        for c in range(COLUMN_COUNT):
            for r in range(ROW_COUNT):
                if board_state[r][c] != 0:
                    # שים לב: אנחנו לא מציירים את הדיסקית החדשה מהמערך עדיין!
                    # כי אנחנו באנימציה שלה כרגע
                    if not (c == col and r == row):
                        assets.draw_piece(screen, c, r, board_state[r][c])

        # ב. ציור הדיסקית הנופלת (האנימציה)
        assets.draw_piece(screen, col, row, player, y_offset=y)

        # ג. ציור הלוח מלמעלה (כדי שהדיסקיות יהיו "בפנים")
        assets.draw_board_overlay(screen)

        pygame.display.update()
        clock.tick(60)  # 60 FPS חלק


def main():
    pygame.init()
    screen = pygame.display.set_mode(size)
    pygame.display.set_caption("Connect 4 Pro")
    myfont = pygame.font.SysFont("monospace", 75)

    assets = AssetManager()
    env = Connect4Env()
    agent = Agent()

    model_path = get_latest_model_path()
    if model_path:
        agent.model.load_state_dict(torch.load(model_path, map_location=agent.device))
        agent.model.eval()
        agent.epsilon = 0

    board = env.reset(show_board=False)

    # ציור ראשוני
    screen.fill(BLACK)
    assets.draw_board_overlay(screen)
    pygame.display.update()

    game_over = False
    turn = 1

    while not game_over:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                sys.exit()

            if event.type == pygame.MOUSEMOTION and turn == 1:
                # ציור דיסקית מרחפת למעלה
                # מנקים רק את החלק העליון
                pygame.draw.rect(screen, BLACK, (0, 0, width, SQUARESIZE))
                posx = event.pos[0]
                # ציור הדיסקית המרחפת
                if assets.use_images:
                    # מרכזים את התמונה סביב העכבר
                    img_x = posx - SQUARESIZE // 2
                    screen.blit(assets.red_img, (img_x, 0))
                else:
                    pygame.draw.circle(screen, (255, 0, 0), (posx, int(SQUARESIZE / 2)), RADIUS)
                pygame.display.update()

            if event.type == pygame.MOUSEBUTTONDOWN and turn == 1:
                pygame.draw.rect(screen, BLACK, (0, 0, width, SQUARESIZE))
                posx = event.pos[0]
                col = int(math.floor(posx / SQUARESIZE))

                if col in env.get_valid_locations():
                    # חישוב השורה הפנויה (בשביל האנימציה)
                    row = 0
                    for r in range(5, -1, -1):
                        if board[r][col] == 0:
                            row = r
                            break

                    # הפעלת האנימציה!
                    animate_drop(screen, assets, board, col, row, 1)

                    # עדכון לוגי
                    next_state, reward, done = env.step(col, 1, show_board=False)
                    board = next_state

                    if done:
                        # ציור סופי ליתר ביטחון
                        assets.draw_piece(screen, col, row, 1)  # לוודא שהיא במקום
                        assets.draw_board_overlay(screen)

                        label = myfont.render("YOU WON!", 1, (255, 0, 0))
                        screen.blit(label, (40, 10))
                        pygame.display.update()
                        game_over = True

                    turn = -1

        # --- תור הסוכן ---
        if turn == -1 and not game_over:
            # אין צורך ב-Wait מלאכותי כי האנימציה לוקחת זמן

            col = agent.act(board, env.get_valid_locations())

            # חישוב השורה הפנויה
            row = 0
            for r in range(5, -1, -1):
                if board[r][col] == 0:
                    row = r
                    break

            # אנימציה לסוכן
            animate_drop(screen, assets, board, col, row, -1)

            next_state, reward, done = env.step(col, -1, show_board=False)
            board = next_state

            if done:
                if check_winner(board, -1):
                    label = myfont.render("AGENT WON!", 1, (255, 255, 0))
                    screen.blit(label, (40, 10))
                else:
                    label = myfont.render("DRAW!", 1, WHITE)
                    screen.blit(label, (40, 10))

                pygame.display.update()
                game_over = True

            turn = 1

        if game_over:
            pygame.time.wait(3000)


if __name__ == "__main__":
    main()