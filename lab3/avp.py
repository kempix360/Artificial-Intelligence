from exceptions import GameplayException
from connect4 import Connect4
from randomagent import RandomAgent
from minmaxagent import MinMaxAgent
from minmaxagent import MinMaxAgent_heuristic

counter_x = 0
counter_o = 0
draws = 0
for i in range(3):
    connect4 = Connect4(width=7, height=6)
    agent1 = MinMaxAgent('x')
    agent2 = MinMaxAgent_heuristic('o')
    while not connect4.game_over:
        connect4.draw()
        try:
            if connect4.who_moves == agent1.my_token:
                n_column = agent1.decide(connect4)
            else:
                n_column = agent2.decide(connect4)
            connect4.drop_token(n_column)
        except (ValueError, GameplayException):
            print('invalid move')
        connect4.draw()
    if connect4.wins == 'x':
        counter_x += 1
    elif connect4.wins is None:
        draws += 1
    elif connect4.wins == 'o':
        counter_o += 1

print("How many times x won: ", counter_x)
print("How many draws: ", draws)
print("How many times o won: ", counter_o)
