import numpy as np
from scipy.signal import convolve2d


class Connect4Env:
    def __init__(self):
        self.board = np.zeros((6, 7), dtype=int)
        self.counter = 0
        self.games = 1
        self.valid_cols = []
        self.player = 1

    def reset(self, show_board=True):
        self.board = np.zeros((6, 7), dtype=int)
        self.counter = 0
        self.valid_cols = []
        if show_board:
            print(self.board, "\n")
        return self.board

    def get_valid_locations(self):
        valid_cols = np.where(self.board[0] == 0)[0]
        return valid_cols

    def step(self, action, player_name, show_board=False):
        done = False
        reward = 0
        self.push_element(action, player_name)
        if show_board:
            self.show_board()
        if check_winner(self.board, player_name):
            done = True
            reward = 10 * player_name
        elif len(self.get_valid_locations()) == 0:
            done = True
            reward = 0
        next_state = self.board

        return next_state, reward, done

    def push_element(self, col, player):
        for r in range(5, -1, -1):
            if self.board[r][col] == 0:
                self.board[r][col] = player
                break

    def show_board(self):
        print(self.board, end="\n")


def check_winner(board, player_val):
    """
    board: 2D numpy array (6x7) with values 0, 1, -1
    player_val: The value to check for (1 or -1)
    """

    # 1. הגדרת ה"פילטרים" (Kernels) שאנו מחפשים
    # כל פילטר מייצג כיוון אחר של רצף באורך 4

    # רצף אופקי (שורה)
    horizontal_kernel = np.array([[1, 1, 1, 1]])

    # רצף אנכי (עמודה)
    vertical_kernel = np.array([[1],
                                [1],
                                [1],
                                [1]])

    # רצף אלכסוני (ראשי)
    diag1_kernel = np.eye(4, dtype=int)

    # רצף אלכסוני (משני)
    diag2_kernel = np.fliplr(diag1_kernel)

    detection_kernels = [horizontal_kernel, vertical_kernel, diag1_kernel, diag2_kernel]

    # 2. מעבר על כל הפילטרים ובדיקה האם יש התאמה
    for kernel in detection_kernels:
        # הפונקציה convolve2d "מריצה" את הפילטר על כל הלוח ומסכמת את החפיפות
        # mode='valid' אומר שלא נבדוק אזורים מחוץ לגבולות הלוח
        conv_result = convolve2d(board, kernel, mode='valid')

        # אם באחד המקומות הסכום שווה ל-4 (או -4), יש לנו מנצח
        # אנו בודקים אם יש ערך ששווה ל- 4 * הערך של השחקן (כלומר 4 או -4)
        if (conv_result == 4 * player_val).any():
            return True

    return False
