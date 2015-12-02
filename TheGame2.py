import GUI as gui
import random
import time
import Player
from tkinter import *


class Tile:
    def __init__(self, value=0):
        self.value = value

    def __str__(self):
        return str(self.value)


class GameRules:
    def __init__(self, size=4):
        self.grid = [[Tile() for i in range(size)] for j in range(size)]
        self.addRandomTile()

    # def playGame(self, input=0):
    #     done = False
    #     while not done:
    #         print(self)
    #         self.move(input)
    #         if self.lost():
    #             print("You have lost")
    #             break
    #         if self.won():
    #             print("You have won")
    #             break

    def lost(self):
        s = len(self.grid) - 1
        b = True
        for i in range(0,len(self.grid)):
            for j in range(0,len(self.grid[i])):
                val = self.grid[i][j].value
                if val == 0:
                    b = False
                    break
                elif i > 0 and self.grid[i - 1][j].value == val:
                    b = False
                    break
                elif j > 0 and self.grid[i][j - 1].value == val:
                    b = False
                    break
                elif i < s and self.grid[i + 1][j].value == val:
                    b = False
                    break
                elif j < s and self.grid[i][j + 1].value == val:
                    b = False
                    break
        return b

    # def won(self):
    #     for i in range(len(self.grid)):
    #         for j in range(len(self.grid[i])):
    #             if self.grid[i][j].value == 2048:
    #                 return True
    #     return False

    def move(self, direction):
        merged = []
        moved = False
        lines = self.rotate(self.grid, direction + 1)
        for line in lines:
            while len(line) and line[-1].value == 0:
                line.pop(-1)
            i = len(line) - 1
            while i >= 0:
                if line[i].value == 0:
                    moved = True
                    line.pop(i)
                i -= 1
            i = 0
            while i < len(line) - 1:
                if line[i].value == line[i + 1].value and not (line[i] in merged or line[i + 1] in merged):
                    moved = True
                    line[i] = Tile(line[i].value * 2)
                    merged.append(line[i])
                    line.pop(i + 1)
                else:
                    i += 1
            while len(line) < len(self.grid):
                line.append(Tile())
        for line in lines:
            if not len(lines):
                line = [Tile() for i in self.grid]
        self.grid = self.rotate(lines, 0 - (direction + 1))
        if moved:
            self.addRandomTile()
            return True
        return False

    def rotate(self, l, num):
        num = num % 4
        s = len(l) - 1
        l2 = []
        if num == 0:
            l2 = l
        elif num == 1:
            l2 = [[None for i in j] for j in l]
            for y in range(len(l)):
                for x in range(len(l[y])):
                    l2[x][s - y] = l[y][x]
        elif num == 2:
            l2 = l
            l2.reverse()
            for i in l:
                i.reverse()
        elif num == 3:
            l2 = [[None for i in j] for j in l]
            for y in range(len(l)):
                for x in range(len(l[y])):
                    l2[y][x] = l[x][s - y]
        return l2

    def addRandomTile(self):
        availableTiles = self.getAvailableTiles()
        findTile = self.findTile(random.choice(availableTiles))
        randomValue = random.random()
        tileValue = 2
        if(randomValue <= 0.1):
            tileValue = 4
        self.grid[findTile[0]][findTile[1]] = Tile(tileValue)

    def getAvailableTiles(self):
        ret = []
        for i in self.grid:
            for j in i:
                if j.value == 0:
                    ret.append(j)
        return ret

    def findTile(self, tile):
        for i in range(len(self.grid)):
            for j in range(len(self.grid[i])):
                if self.grid[i][j] == tile:
                    return i, j

    def getTileArr(self):
        tileArr = []
        for i in range(len(self.grid)):
            for j in range(len(self.grid[i])):
                tileArr.append(self.grid[i][j].value)
        return tileArr

    def getMaxTile(self):
        tileArr = self.getTileArr()
        maxValue = 0
        for i in range(len(tileArr)):
            if(tileArr[i]>maxValue):
                maxValue = tileArr[i]
        return maxValue