import copy
import math

from exceptions import AgentException


class MinMaxAgent:
    def __init__(self, my_token='o'):
        self.my_token = my_token

    def decide(self, connect4):
        if connect4.who_moves != self.my_token:
            score, move = self.minmax(connect4, 0, 4)
        else:
            score, move = self.minmax(connect4, 1, 4)
        return move

    def minmax(self, connect4, x, d):
        if connect4.game_over or d == 0:
            if connect4.wins == self.my_token:
                return 1, 1
            elif connect4.wins is None:
                return 0, 1
            else:
                return -1, 1

        if x == 1:
            best_score = float('-inf')
            best_move = None
            for move in connect4.possible_drops():
                connect4_copy = copy.deepcopy(connect4)
                connect4_copy.drop_token(move)
                score, _ = self.minmax(connect4_copy, 0, d - 1)
                if score > best_score:
                    best_score = score
                    best_move = move
            return best_score, best_move
        else:
            worst_score = float('inf')
            worst_move = None
            for move in connect4.possible_drops():
                connect4_copy = copy.deepcopy(connect4)
                connect4_copy.drop_token(move)
                score, _ = self.minmax(connect4_copy, 1, d - 1)
                if score < worst_score:
                    worst_score = score
                    worst_move = move
            return worst_score, worst_move


class MinMaxAgent_heuristic:
    def __init__(self, my_token='o'):
        self.my_token = my_token

    def decide(self, connect4):
        if connect4.who_moves != self.my_token:
            score, move = self.minmax(connect4, 0, 4)
        else:
            score, move = self.minmax(connect4, 1, 4)
        return move

    def heuristic(self, connect4):
        height = connect4.height
        width = connect4.width
        value = 0

        middle = int(width / 2)
        for i in range(height):
            if connect4.board[i][middle] == self.my_token:
                value += 0.2
            elif connect4.board[i][middle] != self.my_token and connect4.board[i][middle] != '_':
                value -= 0.2

        for four in connect4.iter_fours():
            my_counter = 0
            opponent_counter = 0
            for piece in four:
                if piece == self.my_token:
                    my_counter += 1
                elif piece != self.my_token and piece != '_':
                    opponent_counter += 1

                if my_counter - opponent_counter == 3:
                    value += 0.5
                elif opponent_counter - my_counter == 3:
                    value -= 0.5

        value = value / 10

        return value
    def minmax(self, connect4, x, d):
        if connect4.game_over:
            if connect4.wins == self.my_token:
                return 1, 1
            elif connect4.wins is None:
                return 0, 1
            else:
                return -1, 1

        if d == 0:
            return self.heuristic(connect4), 1

        if x == 1:
            best_score = float('-inf')
            best_move = None
            for move in connect4.possible_drops():
                connect4_copy = copy.deepcopy(connect4)
                connect4_copy.drop_token(move)
                score, _ = self.minmax(connect4_copy, 0, d - 1)
                if score > best_score:
                    best_score = score
                    best_move = move
            return best_score, best_move
        else:
            worst_score = float('inf')
            worst_move = None
            for move in connect4.possible_drops():
                connect4_copy = copy.deepcopy(connect4)
                connect4_copy.drop_token(move)
                score, _ = self.minmax(connect4_copy, 1, d - 1)
                if score < worst_score:
                    worst_score = score
                    worst_move = move
            return worst_score, worst_move
