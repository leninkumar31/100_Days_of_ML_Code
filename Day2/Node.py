import copy as cp

class Node(object):
    def __init__(self,expanded,visited,TotalSimualtionReward,totalNumVisit,TicTacToe,parent):
        self.expanded = expanded
        self.visited = visited
        self.TotalSimualtionReward = TotalSimualtionReward
        self.TotalNumVisit = totalNumVisit
        self.TicTacToe = TicTacToe
        self.Terminal = self.isTerminal()
        self.child = self.get_child()
        self.parent = parent

    def get_child(self):
        child = []
        if self.visited:
            for row in range(self.TicTacToe.board.shape[0]):
                for col in range(self.TicTacToe.board.shape[1]):
                    if self.TicTacToe.board[row][col]==0:
                        childTicTacToe = cp.deepcopy(self.TicTacToe)
                        childTicTacToe.make_move(row,col)
                        child.append(Node(False,False,0,0,childTicTacToe,self))
        return child

    def compareTo(self,board):
        for i in range(len(self.child)):
            cnt = 0
            for row in range(3):
                for col in range(3):
                    if self.child[i].TicTacToe.board[row][col]==board[row][col]:
                        cnt = cnt + 1
            if cnt ==9:
                return self.child[i]
        return None

    def isTerminal(self):
        return self.TicTacToe.GameOver

