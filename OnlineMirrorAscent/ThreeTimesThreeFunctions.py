import numpy as np
import matplotlib.pyplot as plt


def updateStrategy(iterations, payoff1, payoff2, strategy1init, strategy2init, algorithm):
    """ makes one Gradient Ascent with Euclidean Projections or Online Mirror Ascent (with entropic regularizer)
        #update step for every single point in the grid
        Mirror Descent in this case uses entropic regularization: h(x) = sum_i=1^d(x[i] * log(x[i]))

        Args:
            :param iterations: (int): The number of iterations for determining step size eta
            :param payoff1: (np.array)): numpy payoff matrix of row Player (Player1). Use utility maximization
            :param payoff2: (np.array): numpy payoff matrix of column Player (Player2)
            :param strategy1init: (np.array): initial strategy profile for player 1, e.g. np.array([1/9,2/9,6/9])
            :param strategy2init: (np.array): initial strategy profile for player 2
            :param algorithm: (string): choose from:
                -"POGA": projected online gradient ascent, see Bachelor Thesis p.7
                -"EGA": entropic gradient ascent <=> Online Mirror Ascent with entropic regularizer,
                        see Bachelor Thesis p.11

        Returns:
            :return strategy1: (list of np.arrays): stores sequence of strategy profiles for player 1
                                                    according to the algorithm
            :return strategy2: (list of np.arrays): stores sequence of strategy profiles for player 2
                                                    according to the algorithm
    """

    # check if algorithm name is not allowed
    if algorithm != "POGA" and algorithm != "EGA":
        print("Please only use 'POGA' and 'EGA' as algorithm")
        return

    # strategy1, strategy2 are the lists of strategies over time
    strategy1 = [strategy1init]
    strategy2 = [strategy2init]

    # step size
    eta = calculateStepSize(iterations)
    for i in range(iterations):
        strategy1_before = strategy1[-1]
        strategy2_before = strategy2[-1]
        strategy1_after, strategy2_after = updateStep(payoff1, payoff2,
                                                      strategy1_before, strategy2_before, eta, algorithm)
        strategy1 += [strategy1_after]
        strategy2 += [strategy2_after]

    return strategy1, strategy2


def updateStep(payoff1, payoff2, strategy1, strategy2, eta, algorithm):
    """ makes one update step for one specific strategy profile

        Args:
            :param payoff1: numpy payoff matrix of row Player (Player1)
            :param payoff2: numpy payoff matrix of column Player (Player2)
            :param strategy1: Player1: [P(action1), P(action2), P(action3)], e.g. np.array([1/9,2/9,6/9])
            :param strategy2: Player 2 accordingly
            :param eta: stepSize
            :param algorithm: (string): choose from:
                -"POGA": projected online gradient ascent, see Bachelor Thesis p. 7
                -"EGA": entropic gradient ascent, see Bachelor Thesis p. 11

        Returns:
            :return strategy1_after: (np.array): strategy profile of player 1 after one step
            :return strategy2_after: (np.array): strategy profile of player 2 after one step
        """

    # compute gradient
    grad1 = payoff1.dot(strategy2)
    grad2 = strategy1.transpose().dot(payoff2)

    # projected online gradient ascent
    if algorithm == "POGA":
        strategy1_after = strategy1 + eta * grad1
        strategy1_after = project(strategy1_after)

        strategy2_after = strategy2 + eta * grad2
        strategy2_after = project(strategy2_after)

    # entropic gradient ascent
    else:
        strategy2_after = strategy2 * np.exp(eta * grad2) / sum(strategy2 * np.exp(eta * grad2))
        strategy1_after = strategy1 * np.exp(eta * grad1) / sum(strategy1 * np.exp(eta * grad1))

    return strategy1_after, strategy2_after


def project(v):
    """ borrowed from Mathieu Blondel (https://gist.github.com/mblondel/6f3b7aaad90606b98f71)
        take a vector of three dimensions and output the euclidean projection onto the simplex

    Args:
        :param: v: (array): the vector to be projected

    Return:
        :return: w: (array): the projected vector

    """
    n_features = v.shape[0]
    u = np.sort(v)[::-1]
    cssv = np.cumsum(u) - 1
    ind = np.arange(n_features) + 1
    cond = u - cssv / ind > 0
    rho = ind[cond][-1]
    theta = cssv[cond][-1] / float(rho)
    w = np.maximum(v - theta, 0)
    return w


def calculateStepSize(iterations):
    """ NOTE: I didnt tune the step Size to each specific game, but use some constant step size instead.

    for online mirror descent step size should be: eta = sqrt(log(d))/(L*sqrt(2T)) (Shalev-Shwartz, p.140)
        where,
        d: dimensions (number of strategies)
        lip: Lipschitz constance
        iterations: number of Iterations
        this eta yields regret bound of: Regret_T(S) <= B*L*sqrt(2*log(d)*T), where S = {x : l1norm(x) = B AND x > 0}
        note: This is better than Gradient Descent using Euclidean Regularizer

        Args:
            :param iterations: (int):  number of Iterations that the algorithm should run

        Returns:
            (float): stepSize eta
    """

    # number of strategies
    d = 2
    # Lipschitz Constant
    lip = 1
    # step size
    # eta = np.sqrt(np.log(d)) / (lip * np.sqrt(2 * iterations))

    # using constant step size
    eta = 0.1
    return eta


def plot(iterations, strategy1, strategy2, gameName, strategyNames, algorithm, figsize=6, dpi=100):
    """ will plot a line graph using matplotlib according to the sequence of play for each player

         Args:
            :param strategyNames: (array of strings): Strategy names according to syntax:
                                strategyNames = [nameOfStrategy1Action1,nameOfStrategy1Action2, nameOfStrategy1Action3,
                                                nameOfStrategy2Action1, nameOfStrategy2Action2, nameOfStrategy2Action3]
            :param iterations: number of iterations played
            :param strategy1: list of sequence of strategies played by player 1
            :param strategy2: list of sequence of strategies played by player 2
            :param gameName: (string): The name of the game that will be displayed as title
            :param algorithm: (string): choose from:
                -"POGA": projected online gradient ascent, see Bachelor Thesis p.7
                -"EGA": entropic gradient ascent, see Bachelor Thesis p.11
            :param figsize: optional size of figures
            :param dpi: optional resolution of plot

    """

    plt.figure(figsize=(figsize, figsize), dpi=dpi)

    labelPlot(gameName, algorithm)

    # time on x axis
    x = range(iterations + 1)

    # extract probability paths, y values
    p11 = [p[0] for p in strategy1]
    p12 = [p[1] for p in strategy1]
    p13 = [p[2] for p in strategy1]

    p21 = [p[0] for p in strategy2]
    p22 = [p[1] for p in strategy2]
    p23 = [p[2] for p in strategy2]

    # plot lines
    plt.plot(x, p11, label='P(' + strategyNames[0] + ') Player 1', c='#0065BD', ls='-')
    plt.plot(x, p12, label='P(' + strategyNames[1] + ') Player 1', c='#0065BD', ls='--')
    plt.plot(x, p13, label='P(' + strategyNames[2] + ') Player 1', c='#0065BD', ls=':')
    plt.plot(x, p21, label='P(' + strategyNames[3] + ') Player 2', c='#A2AD00', ls='-')
    plt.plot(x, p22, label='P(' + strategyNames[4] + ') Player 2', c='#A2AD00', ls='--')
    plt.plot(x, p23, label='P(' + strategyNames[5] + ') Player 2', c='#A2AD00', ls=':')

    plt.ylim(-0.05, 1.05)
    plt.xlim(-0.05, iterations + 1)

    plt.legend()
    plt.show()
    return


def labelPlot(gameName, algorithm):
    if algorithm == "POGA":
        algorithmName = "projected online gradient ascent"
    elif algorithm == "EGA":
        algorithmName = "entropic gradient ascent"
    else:
        algorithmName = ""
    title = gameName + ' \n ' + '(' + algorithmName + ')'
    plt.ylabel('$x_{i,a}$', fontsize=15)
    plt.xlabel('time $t$', fontsize=10)
    plt.title(title, fontsize=14)
    return


def randomPlots(numberOfPlots, gameName, strategyNames, iterations,
                payoff1, payoff2, algorithm, figsize=6, dpi=100):
    """ generates random simplex vectors as initial strategies for both players and runs the algorithm

    :param numberOfPlots: number of plots that the function will plot. Be aware of memory constraint
    :param gameName: as above
    :param strategyNames: as above
    :param iterations: as above
    :param payoff1: as above
    :param payoff2: as above
    :param algorithm: as above
    :param figsize: optional figure size
    :param dpi: optional dpi
    """

    for i in range(numberOfPlots):
        strategy1init = randomSimplexVector()
        strategy2init = randomSimplexVector()
        strategy1, strategy2 = updateStrategy(iterations, payoff1, payoff2, strategy1init, strategy2init, algorithm)
        plot(iterations, strategy1, strategy2, gameName, strategyNames, algorithm, figsize, dpi)


def systematicPlots(n_per_dim, gameName, strategyNames,
                    iterations, payoff1, payoff2, algorithm, figsize=6, dpi=100):
    """ generates strategy vectors systematically and then plots each combination of initial strategies.
        Doesnt apply so well for EGA as function starts with boundary strategies

    :param n_per_dim: granularity of probability vector
    :param gameName: as above
    :param strategyNames: as above
    :param iterations: as above
    :param payoff1: as above
    :param payoff2: as above
    :param algorithm: as above
    :param figsize: optional figure size
    :param dpi: optional dpi
    """

    strategies1 = systematicSimplexVector(n_per_dim)
    strategies2 = systematicSimplexVector(n_per_dim)

    for i in range(len(strategies1)):
        for j in range(len(strategies2)):
            strategy1init = strategies1[i]
            strategy2init = strategies2[j]
            strategy1, strategy2 = updateStrategy(iterations, payoff1, payoff2, strategy1init, strategy2init, algorithm)
            plot(iterations, strategy1, strategy2, gameName, strategyNames, algorithm, figsize, dpi)


def calculateFrequencies(pInit, iterations, payoff1, payoff2, algorithm):
    """ calculates the empirical frequency distribution given an algorithm

    :param pInit: array of 6 floats. First 3 denote the initial probability distribution of player 1, play2 accordingly
    :param iterations: number of iterations (time)
    :param payoff1: payoff matrix player 1
    :param payoff2: payoff matrix player 2
    :param algorithm: (string): choose from:
                -"POGA": projected online gradient ascent, see Bachelor Thesis p. 7
                -"EGA": entropic gradient ascent, see Bachelor Thesis p. 11

    :return: frequencies: array of 6 frequency lists for each action
    """

    numStrategy1Action1Played, numStrategy1Action2Played, numStrategy1Action3Played, \
    numStrategy2Action1Played, numStrategy2Action2Played, numStrategy2Action3Played = np.zeros(6)

    frequencyStrategy1Action1, frequencyStrategy1Action2, frequencyStrategy1Action3, \
    frequencyStrategy2Action1, frequencyStrategy2Action2, frequencyStrategy2Action3 = [[], [], [], [], [], []]

    eta = calculateStepSize(iterations)

    strategy1 = np.array([pInit[0], pInit[1], pInit[2]])
    strategy2 = np.array([pInit[0], pInit[1], pInit[2]])

    for t in range(iterations):

        max1 = max(strategy1)
        max2 = max(strategy2)
        maxIndex1 = np.where(strategy1 == max1)
        maxIndex2 = np.where(strategy2 == max2)
        maxIndex1Any = maxIndex1[0][0]
        maxIndex2Any = maxIndex2[0][0]

        if maxIndex1Any == 0:
            numStrategy1Action1Played += 1
        elif maxIndex1Any == 1:
            numStrategy1Action2Played += 1
        elif maxIndex1Any == 2:
            numStrategy1Action3Played += 1

        if maxIndex2Any == 0:
            numStrategy2Action1Played += 1
        elif maxIndex2Any == 1:
            numStrategy2Action2Played += 1
        elif maxIndex2Any == 2:
            numStrategy2Action3Played += 1

        frequencyStrategy1Action1 += [numStrategy1Action1Played / (t + 1)]
        frequencyStrategy1Action2 += [numStrategy1Action2Played / (t + 1)]
        frequencyStrategy1Action3 += [numStrategy1Action3Played / (t + 1)]
        frequencyStrategy2Action1 += [numStrategy2Action1Played / (t + 1)]
        frequencyStrategy2Action2 += [numStrategy2Action2Played / (t + 1)]
        frequencyStrategy2Action3 += [numStrategy2Action3Played / (t + 1)]

        strategy1, strategy2 = updateStep(payoff1, payoff2, strategy1, strategy2, eta, algorithm)

    frequencies = [frequencyStrategy1Action1, frequencyStrategy1Action2, frequencyStrategy1Action3, \
                   frequencyStrategy2Action1, frequencyStrategy2Action2, frequencyStrategy2Action3]

    return frequencies


def plotFrequencies(iterations, frequencies, strategyNames, gameName, algorithm):
    """ plots the empirical frequency of play for all strategies played

    :param iterations: number of iterations ()time
    :param frequencies: array of 6 lists generated by calculateFrequencies()
    :param strategyNames: array of 6 strings
    :param gameName: string
    :param algorithm: (string): choose from:
                -"POGA": projected online gradient ascent, see Bachelor Thesis p. 7
                -"EGA": entropic gradient ascent, see Bachelor Thesis p. 11

    """
    plt.figure(figsize=(6, 6), dpi=100)
    labelTitle(gameName, algorithm)

    # time on x axis
    x = range(iterations)

    plt.plot(x, frequencies[0], label='Player 1: ' + strategyNames[0], c='#0065BD', ls='-')
    plt.plot(x, frequencies[1], label='Player 1: ' + strategyNames[1], c='#0065BD', ls='--')
    plt.plot(x, frequencies[2], label='Player 1: ' + strategyNames[2], c='#0065BD', ls=':')

    plt.plot(x, frequencies[0], label='Player 2: ' + strategyNames[0], c='#A2AD00', ls='-')
    plt.plot(x, frequencies[1], label='Player 2: ' + strategyNames[1], c='#A2AD00', ls='--')
    plt.plot(x, frequencies[2], label='Player 2: ' + strategyNames[2], c='#A2AD00', ls=':')

    plt.ylim(-0.05, 1.05)
    plt.xlabel('time $t$', fontsize=10)
    plt.ylabel('frequency', fontsize=10)
    plt.legend(loc='best', fontsize=10)
    plt.show()


def labelTitle(gameName, algorithm):
    """ Gives plot a title depending on the algorithm and the gameName provided in initialization

    """
    if algorithm == "POGA":
        algorithmName = "projected online gradient ascent"
    elif algorithm == "EGA":
        algorithmName = "entropic gradient ascent"
    else:
        algorithmName = ""
    title = gameName + ' \n ' + '(' + algorithmName + ')'
    plt.title(title, fontsize=14)


def randomSimplexVector():
    """ Return uniformly random vector in the 3-simplex """

    k = np.random.exponential(scale=1.0, size=3)
    return np.array(k / sum(k))


def systematicSimplexVector(n_per_dim):
    """ generates 3d probability vectors in a systematicWay
    borrowed from https://stackoverflow.com/questions/51957207/appropriate-numpy-scipy-function-to-
                        interpolate-function-defined-on-simplex-non

    :param n_per_dim:
    :return:
    """
    xlist = np.linspace(0.0, 1.0, n_per_dim)
    ylist = np.linspace(0.0, 1.0, n_per_dim)
    zlist = np.linspace(0.0, 1.0, n_per_dim)
    return np.array([[x, y, z] for x in xlist for y in ylist for z in zlist
                     if np.allclose(x + y + z, 1.0)])


def isFeasible(strategy):
    return sum(strategy) == 1 and min(strategy) >= 0 and max(strategy) <= 1
