from __future__ import print_function
import numpy as np

class TicTacToe(object):
    def __init__(self):
        self.board = np.zeros((3,3),dtype=int)
        self.GameOver = False
        self.moveCnt = 0
        self.draw = False

    def check_move(self,x,y):
        return x >= 0 and y >= 0 and x < 3 and y < 3 and self.board[x][y]==0

    def check_rows(self):
        for row in range(3):
            if (self.board[row][0]==1 and self.board[row][1]==1 and self.board[row][2]==1) \
                    or (self.board[row][0]==2 and self.board[row][1]==2 and self.board[row][2]==2):
                self.GameOver = True
                break
        return self.GameOver

    def check_cols(self):
        for col in range(3):
            if (self.board[0][col] == 1 and self.board[1][col] == 1 and self.board[2][col] == 1) \
                    or (self.board[0][col] == 2 and self.board[1][col] == 2 and self.board[2][col] == 2):
                self.GameOver = True
                break
        return self.GameOver

    def check_diags(self):
        self.GameOver = (self.board[0][0]==1 and self.board[1][1]==1 and self.board[2][2]==1) \
                        or (self.board[0][0]==2 and self.board[1][1]==2 and self.board[2][2]==2) or \
                        (self.board[0][2]==1 and self.board[1][1]==1 and self.board[2][0]==1) or \
                        (self.board[0][2]==2 and self.board[1][1]==2 and self.board[2][0]==2)
        return self.GameOver

    def isDraw(self):
        if not self.GameOver and self.moveCnt==9:
            self.draw = True;
            self.GameOver = True
        return self.GameOver

    def isGameOver(self):
        return self.check_rows() or self.check_cols() or self.check_diags() or self.isDraw()

    def make_move(self,x,y):
        assert self.check_move(x,y),'You Gave wrong input. Please Choose your move again!!'
        self.board[x][y] = (self.moveCnt % 2) + 1
        self.moveCnt = self.moveCnt + 1
        return self.isGameOver()

    def print_board(self):
        for row in range(self.board.shape[0]):
            print("|",end='')
            for col in range(self.board.shape[1]):
                if self.board[row][col]==0:
                    print(".",sep='|',end='|')
                elif self.board[row][col]==1:
                    print("O",sep='|',end='|')
                else:
                    print("X",sep='|',end='|')
            print()

