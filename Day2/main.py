from __future__ import print_function
from Tic_Tac_Toe import TicTacToe
from Node import Node
import MCTS
import copy as cp

def main():
    BoardObj = TicTacToe()
    currNode = Node(expanded=False, visited=True, TotalSimualtionReward=0, totalNumVisit=1, TicTacToe=BoardObj,parent=None)
    print("Initial Board setting")
    currNode.TicTacToe.print_board()
    while not currNode.Terminal:
        if currNode.TicTacToe.moveCnt & 1:
            x = int(raw_input('Enter row position\n'))
            y = int(raw_input('Enter column position\n'))
            TicTacToeObj = cp.deepcopy(currNode.TicTacToe)
            try:
                TicTacToeObj.make_move(x,y)
            except:
                continue
            nextNode = currNode.compareTo(TicTacToeObj.board)
            if nextNode is None:
                nextNode = Node(expanded=False, visited=True, TotalSimualtionReward=0, totalNumVisit=1, TicTacToe=TicTacToeObj,parent=None)
        else:
            nextNode = MCTS.MonteCarloTreeSearch(currNode, 0.1)
        print("After {} Move".format(nextNode.TicTacToe.moveCnt))
        print(nextNode.TotalSimualtionReward)
        print(nextNode.TotalNumVisit)
        nextNode.TicTacToe.print_board()
        currNode = nextNode
    if currNode.TicTacToe.draw:
        print("Match is Drawn")
    else:
        if currNode.TicTacToe.moveCnt & 1:
            print("First Player won")
        else:
            print("Second Player won")

if __name__=="__main__":
    main()
