from __future__ import print_function
from Tic_Tac_Toe import TicTacToe
from Node import Node
import MCTS
import copy as cp
import pickle
import numpy as np
def create_string(temp):
    board = ''
    for i in range(3):
        for j in range(3):
            board += str(temp[i][j])
    return board
# take action based on state
def take_action(board,Value,player):
    possible_actions = []
    for i in range(3):
        for j in range(3):
            if board[i][j]==0:
                possible_actions.append((i,j))
    # A = np.ones(len(possible_actions),dtype=float)*eps/len(possible_actions)
    action_values = []
    for action in possible_actions:
        board[action[0]][action[1]] = player
        if player & 1:
            action_values.append(Value[create_string(board)])
        else:
            action_values.append(1.0 - Value[create_string(board)])
        board[action[0]][action[1]] = 0
    # if np.min(action_values)==np.max(action_values):
    #     best_action = np.random.randint(len(action_values))
    # else:
    #     best_action = np.argmax(action_values)
    # A[best_action] += 1.0 - eps
    # print(A)
    return possible_actions[np.argmax(action_values)]

def main():
    mcts_wins = 0
    draws = 0
    mcts_loses = 0
    for game in range(100):
        VF_pickle = open("/Users/lpothabattula/Desktop/TensorFlow/Day4/TicTacToeValFun.pickle", "rb")
        ValueFunction = pickle.load(VF_pickle)
        BoardObj = TicTacToe()
        currNode = Node(expanded=False, visited=True, TotalSimualtionReward=0, totalNumVisit=1, TicTacToe=BoardObj,parent=None)
        # print("Initial Board setting")
        # currNode.TicTacToe.print_board()
        while not currNode.Terminal:
            player = currNode.TicTacToe.moveCnt % 2 + 1
            if currNode.TicTacToe.moveCnt & 1:
                x,y = take_action(currNode.TicTacToe.board,Value=ValueFunction,player=player)
                TicTacToeObj = cp.deepcopy(currNode.TicTacToe)
                TicTacToeObj.make_move(x,y)
                nextNode = currNode.compareTo(TicTacToeObj.board)
                if nextNode is None:
                    nextNode = Node(expanded=False, visited=True, TotalSimualtionReward=0, totalNumVisit=1, TicTacToe=TicTacToeObj,parent=None)
            else:
                nextNode = MCTS.MonteCarloTreeSearch(currNode, 0.1)
            # print("After {} Move".format(nextNode.TicTacToe.moveCnt))
            # print(nextNode.TotalSimualtionReward)
            # print(nextNode.TotalNumVisit)
            # nextNode.TicTacToe.print_board()
            currNode = nextNode
        if currNode.TicTacToe.draw:
            draws += 1
            print("Match {}:Drawn".format(game))
        else:
            if currNode.TicTacToe.moveCnt & 1:
                mcts_wins += 1
                print("Match {}:First Player won".format(game))
            else:
                mcts_loses += 1
                print("Match {}:Second Player won".format(game))
    print("Final analysis:MCTS vs TD")
    print("MCTS won {} times".format(mcts_wins))
    print("match drawn {} times".format(draws))
    print("MCTS lost {} times".format(mcts_loses))
if __name__=="__main__":
    main()
