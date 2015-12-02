from tkinter import *
from sympy import stats
from math import log, ceil
import ANN
import Player
from scipy import stats
import random, time
import GUI
import TheGame2 as GAME
import theano.tensor.nnet as Tann
import theano.tensor as T
import ai2048demo

class Runner:
    """
    Creates and runs a player for 2048
    """
    def __init__(self, randomPlayer=False, plannedGames=50, animationTime=1, layer_sizes=[16, 250, 4], lr=.1,
                 activation=ANN.relu, max_iterations=500, rand_limit_min=-.02, rand_limit_max=.02,
                 learning_set="learningset_prettysmall", test_set="testset1"):
        self.window = GUI.GameWindow()
        self.player = Player.Player(randomPlayer=randomPlayer, layer_sizes=layer_sizes, lr=lr, activation=activation,
                                    max_iterations=max_iterations, rand_limit_min=rand_limit_min, rand_limit_max=rand_limit_max,
                                    learning_set=learning_set, test_set=test_set)
        self.game = GAME.GameRules()
        self.numGames = 0
        self.randomcounter = 0.0
        self.movecounter = 0.0
        self.plannedGames = plannedGames
        self.animationTime = animationTime
        self.values = []
        self.randomvalues = []

    def getAvg(self, list):
        """
        Gets an average of a list
        """
        return sum(list)/len(list)

    def reset(self):
        """
        Resets the game
        """
        self.game = GAME.GameRules()
        self.window.reset()
        self.movecounter = 0.0
        self.randomcounter = 0.0

    def runGame(self):
        """
        Starts the game
        """
        self.animate()

    def drawMove(self):
        """
        Moves the board
        :return:
        """
        direction = self.player.move(boardValues=self.game.getTileArr())
        for i in range(len(direction)):
            moved = self.game.move(direction[i])
            if moved:
                break

        self.movecounter += 1.0
        boardArr = self.game.getTileArr()
        self.window.update_view(boardArr)

    def animate(self):
        """
        Plays the game
        """
        if not self.game.lost():
            self.drawMove()
            root.after(self.animationTime, self.animate)
        else:
            self.numGames += 1
            self.values.append(self.game.getMaxTile())
            if not self.movecounter == 0.0: print("Max value:", self.game.getMaxTile(), "Random percentage:", self.randomcounter/self.movecounter)
            self.reset()
            if self.numGames < self.plannedGames:
                root.after(self.animationTime, self.animate)
            else:
                print(self.values)
                print("Avg:",self.getAvg(self.values))
                root.quit()

def ttest(ai_values, random_values):
    """
    Prints information from a Welch T-test for two lists
    """
    t, p = stats.ttest_ind(ai_values, random_values, equal_var=False)
    print(p)
    print(t)
    print(-log(p, 10))
    final_score = max(0,min(7, ceil(-log(p, 10))))
    print("Grade: ", final_score)


# Trains one ANN and one random player, and plays 50 games each
start = time.time()
root = Tk()
runner = Runner(randomPlayer=False, plannedGames=50, animationTime=1, layer_sizes=[16, 100, 100, 4], lr=.5,
                 activation=ANN.relu, max_iterations=10, rand_limit_min=-.02, rand_limit_max=.02,
                learning_set="another2048_j", test_set="another2048_j")
runner.runGame()
aiValues = runner.values
root.mainloop()

root = Tk()
print("Doing random player...")
randomPlayer = Runner(randomPlayer=True, plannedGames=50, animationTime=1)
randomPlayer.runGame()
randomValues = randomPlayer.values
root.mainloop()

end = time.time()
print("Time:", end - start)
print("Avg AI:", runner.getAvg(aiValues), ", random player:", randomPlayer.getAvg(randomValues))