from __future__ import print_function
import numpy as np
from TicTacToe import TicTacToe
from collections import defaultdict

def softmax(x):
    z = np.exp(x)
    return z/np.sum(z)

def create_string(temp):
    board = ''
    for i in range(3):
        for j in range(3):
            board += str(temp[i][j])
    return board

def take_action(board,Value,player,eps,isTrain):
    possible_actions = []
    for i in range(3):
        for j in range(3):
            if board[i][j]==0:
                possible_actions.append((i,j))
    A = np.ones(len(possible_actions),dtype=float)*eps/len(possible_actions)
    action_values = []
    for action in possible_actions:
        board[action[0]][action[1]] = player
        if isTrain or player & 1:
            action_values.append(Value[create_string(board)])
        else:
            action_values.append(1.0 - Value[create_string(board)])
        board[action[0]][action[1]] = 0
    if np.min(action_values)==np.max(action_values):
        best_action = np.random.randint(len(action_values))
    else:
        best_action = np.argmax(action_values)
    A[best_action] += 1.0 - eps
    return possible_actions[np.random.choice(range(len(possible_actions)),p=A)]

# initializing value function
Value = defaultdict(float)
UpdateCnt = defaultdict(float)
def init(board):
    if board.GameOver:
        if board.draw:
            Value[create_string(board.board)] = 0.0
        else:
            if board.moveCnt % 2==0:
                Value[create_string(board.board)] = 0.0
            else:
                Value[create_string(board.board)] = 1.0
    else:
        Value[create_string(board.board)] = 0.5
        for i in range(3):
            for j in range(3):
                if board.board[i][j]==0:
                    board.make_move(i,j)
                    init(board)
                    board.unmake_move(i,j)
    UpdateCnt[create_string(board.board)] = 0
init(TicTacToe())
print(len(Value))
epsilon = 0.2
alpha = 0.9
episodes = 50000
for _ in range(episodes):
    board = TicTacToe()
    curr_state = create_string(board.board)
    UpdateCnt[curr_state] += 1
    while not board.GameOver:
        player = board.moveCnt%2 + 1
        action = take_action(board.board,Value,player,epsilon,isTrain=True)
        board.make_move(action[0],action[1])
        next_state = create_string(board.board)
        Value[curr_state] += alpha * (Value[next_state]-Value[curr_state])
        UpdateCnt[next_state] += 1
        # print(curr_state,Value[curr_state])
        # print(next_state,Value[next_state])
        curr_state = next_state
for _ in range(100):
    BoardObj = TicTacToe()
    print("Initial Board setting")
    BoardObj.print_board()
    while not BoardObj.GameOver:
        print("probability of winning from this state is {}".format(Value[create_string(BoardObj.board)]))
        print("{} many times we updated this state".format(UpdateCnt[create_string(BoardObj.board)]))
        player = BoardObj.moveCnt % 2 + 1
        if player%2==0:
            x = int(raw_input('Enter row position\n'))
            y = int(raw_input('Enter column position\n'))
            try:
                BoardObj.make_move(x,y)
            except:
                print("Entered wrong details")
                continue
        else:
            x,y = take_action(BoardObj.board,Value,player,eps=epsilon,isTrain=True)
            BoardObj.make_move(x, y)
        print("After {} Move".format(BoardObj.moveCnt))
        BoardObj.print_board()
    if BoardObj.draw:
        print("Match is Drawn")
    else:
        if BoardObj.moveCnt & 1:
            print("First Player won")
        else:
            print("Second Player won")
