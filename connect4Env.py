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

    horizontal_kernel = np.array([[1, 1, 1, 1]])

    vertical_kernel = np.array([[1],
                                [1],
                                [1],
                                [1]])

    diag1_kernel = np.eye(4, dtype=int)

    diag2_kernel = np.fliplr(diag1_kernel)

    detection_kernels = [horizontal_kernel, vertical_kernel, diag1_kernel, diag2_kernel]

    for kernel in detection_kernels:
        conv_result = convolve2d(board, kernel, mode='valid')

        if (conv_result == 4 * player_val).any():
            return True

    return False
