import numpy as np
import matplotlib.pyplot as plt

colors = ['#0065BD', '#A2AD00', 'b', '#E37222']


def updateStrategy(iterations, gridSize, payoff1, payoff2, algorithm):
    """ makes one Gradient Ascent with Euclidean Projections or Online Mirror Ascent (with entropic regularizer)
        #update step for every single point in the grid
        Mirror Descent in this case uses entropic regularization: R(x) = sum_i=1^d(x[i] * log(x[i]))
        which is 1 strongly convex over the probability simplex (Shalev-Shwartz, p.136))

        Args:
            :param payoff2: (int): The number of iterations for determining step size eta
            :param payoff1: (int): number vor initial probability pairs. Number of vector will be gridSize^2
            :param gridSize: (np.array)): numpy payoff matrix of row Player (Player1). Use utility maximization
            :param iterations: (np.array): numpy payoff matrix of column Player (Player2)
            :param algorithm: (string): choose from:
                -"POGA": projected online gradient ascent, see Shalev-Swartz p.144
                -"EGA": entropic gradient ascent <=> Online Mirror Ascent with entropic regularizer, see Shalev-Swartz p.144

        Returns:
            (np.matrix): diff1 describes how much the probability for Player 1
                to play strategy 1 has changed for all girdCells
            (np.matrix): diff2 describes how much the probability for Player 2
                to play strategy 1 has changed for all girdCells
    """

    # check if algorithm is allowed
    if algorithm != "POGA" and algorithm != "EGA":
        print("Please only use 'POGA' and 'EGA' as algorithm")
        return

    # store difference in strategies after one update step for player 1, player 2
    eta = calculateStepSize(iterations)

    # P1(Strategy[0]), P2(Strategy[0]) // note that EGA is not defined on simplex edges
    if algorithm == "POGA":
        p1 = p2 = np.linspace(0, 1, gridSize)
    else:
        p1 = p2 = np.linspace(0.001, 0.999, gridSize)
    diff1 = np.zeros((gridSize, gridSize))
    diff2 = np.zeros((gridSize, gridSize))

    for i in range(gridSize):
        for j in range(gridSize):
            # current probabilities x,y
            x = p1[i]
            y = p2[j]

            # current strategy
            strategy1_before = np.array([x, 1 - x])
            strategy2_before = np.array([y, 1 - y])

            strategy1_after, strategy2_after = updateStep(payoff1,
                                                          payoff2,
                                                          strategy1_before,
                                                          strategy2_before,
                                                          eta,
                                                          algorithm)

            # current difference
            diff1_temp = strategy1_after - strategy1_before
            diff2_temp = strategy2_after - strategy2_before

            # store result in difference matrix
            diff1[j][i] = diff1_temp[0]
            diff2[j][i] = diff2_temp[0]

    return diff1, diff2


def updateStep(payoff1, payoff2, strategy1, strategy2, eta, algorithm):
    """ makes one Online Mirror Ascent update step for one specific strategy profile

        Args:
            :param payoff1: numpy payoff matrix of row Player (Player1)
            :param payoff2: numpy payoff matrix of column Player (Player2)
            :param strategy1: Player1: [P(action1), 1-P(action1],
            :param strategy2: Player 2 accordingly
            :param eta: stepSize
            :param algorithm: (string): choose from:
                -"POGA": projected online gradient ascent, see Shalev-Swartz p.144
                -"EGA": entropic gradient ascent, see Shalev-Swartz p.144

        Returns:
            (np.array): strategy profile of player 1 after one step
            (np.array): strategy profile of player 2 after one step
        """

    # compute gradient
    grad1 = payoff1.dot(strategy2)
    grad2 = strategy1.transpose().dot(payoff2)

    # Note this is projected online gradient ascent
    if algorithm == "POGA":
        strategy1_after = strategy1 + eta * grad1
        if not isFeasible(strategy1_after):
            strategy1_after = project(strategy1_after)
        strategy2_after = strategy2 + eta * grad2
        if not isFeasible(strategy2_after):
            strategy2_after = project(strategy2_after)
    else:
        # update strategy (Online Mirror Ascent with entropic regularizer <=> entropic gradient ascent)
        strategy2_after = strategy2 * np.exp(eta * grad2) / sum(strategy2 * np.exp(eta * grad2))
        strategy1_after = strategy1 * np.exp(eta * grad1) / sum(strategy1 * np.exp(eta * grad1))

    return strategy1_after, strategy2_after


def plot(gridSize, diff1, diff2, pne, mne, gameName, nameOfStrategy1, nameOfStrategy2, algorithm, plotType,
         payoff1=np.array([]), payoff2=np.array([]), stableTo=None):
    """ will plot a vector field using matplotlib

         Args:
            :param gridSize: (int): number of discrete points to calculate plot a vector
            :param diff1: (np.array): matrix that measures difference of probability
                that Player 1 plays Strategy1 1 after one updateStep (x direction)
            :param diff2: (np.array): matrix that measures difference of probability
                that Player 2 plays Strategy 1 after one updateStep (y direction)
            :param pne: (list of lists): array of all Pure Nash Equilibria. Syntax [[1,1],[0,0],...]
            :param mne: (list of lists): array of all Mixed Nash Equilibria.
                Syntax (..., (P1(strategy1),P2(strategy1)), ... ), e.g [[3/5,2/5]]
            :param gameName: (string): The name of the game that will be displayed as title
            :param nameOfStrategy1: (string): The name that the first Strategy of player 1 displayed on x axis.
            :param nameOfStrategy2: (string): The name that the first Strategy of player 2 displayed on y axis.
            :param algorithm: (string): choose from:
                -"POGA": projected online gradient ascent, see Shalev-Swartz p.144
                -"EGA": entropic gradient ascent, see Shalev-Swartz p.144
            :param plotType:(string): choose from:
                    -"quiver": creates a vector field
                    -"stream",: creates a streamPlot
            :param payoff1: (np.array): optional argument, only needed when stable points should be plotted
            :param payoff2: (np.array): same as payoff1
            :param stableTo: (list of lists): an optional argument. When not empty the region of stable point
                                            w.r.t to the given nash equilibria will be plotted
    """

    if stableTo is None:
        stableTo = []
    x, y = createMeshGrid(gridSize)
    # create vector field
    plt.figure(figsize=(6, 6), dpi=100)
    # plot all Pure and Mixed Equilibria
    plotEquilibria(pne, mne)
    labelPlot(gameName, nameOfStrategy1, nameOfStrategy2, algorithm)
    legend(pne, mne)
    adjustPlot()
    # depending on type ("quiver" or "stream" show the according plot)
    if plotType == "quiver":
        plt.quiver(x, y, diff1, diff2, pivot="middle", color="#000000")
    elif plotType == "stream":
        plt.streamplot(x, y, diff1, diff2, density=1.5, color="#000000")
    else:
        print("type " + plotType + " is not allowed. only use 'quiver' or 'stream'")
    plotStableRegions(payoff1, payoff2, stableTo)
    plt.show()


def trajectory(p1Init, p2Init, iterations, payoff1, payoff2, algorithm):
    """ calculate not just one step but the whole trajectory of the Online Mirror Ascent algorithms for T iterations

        Args:
            :param p1Init: (float): initial probability that player 1 plays strategy 1
            :param p2Init: (float): initial probability that player 2 plays strategy 1
            :param iterations: (int): number of iterations that the algorithm should go
            :param payoff1: (np.array): payoff matrix for player 1 (utility maximization)
            :param payoff2: (np.array): payoff matrix for player 2 (utility maximization)
            :param algorithm: (string): choose from:
                -"POGA": projected online gradient ascent, see Shalev-Swartz p.144
                -"EGA": entropic gradient ascent, see Shalev-Swartz p.144

        Returns:
            (list of floats): list of probabilities for player 1 to play strategy 1 according to timeStep t
            (list of floats): list of probabilities for player 2 to play strategy 1 according to timeStep t

    """
    eta = calculateStepSize(iterations)
    strategy1 = np.array([p1Init, 1 - p1Init])
    strategy2 = np.array([p2Init, 1 - p2Init])

    # list to store results
    p1, p2 = [p1Init], [p2Init]

    for t in range(iterations):
        # update strategy
        strategy1, strategy2 = updateStep(payoff1, payoff2, strategy1, strategy2, eta, algorithm)

        # store p_Head in result vector p
        p1 += [strategy1[0]]
        p2 += [strategy2[0]]

    return p1, p2


def plotTrajectory(p1Init, p2Init, p1, p2, pne, mne,
                   gameName, nameOfStrategy1, nameOfStrategy2, algorithm,
                   payoff1=np.array([]), payoff2=np.array([]), stableTo=None):
    """ plots the trajectory of a the algorithm for one specific initial strategy

    Args:

        :param p1Init: (float): initial probability that player 1 plays strategy 1. Will be plotted as point.
        :param p2Init: (float): initial probability that player 2 plays strategy 1
        :param p1: (list of floats): list of probabilities for player 1 to play strategy 1 according to timeStep t
        :param p2: (list of floats): list of probabilities for player 2 to play strategy 1 according to timeStep t
        :param pne: (list of lists): array of all Pure Nash Equilibria. Syntax [[1,1],[0,0],...]
        :param mne: (list of lists): array of all Mixed Nash Equilibria.
            Syntax (..., (P1(strategy1),P2(strategy1)), ... ), e.g [[3/5,2/5]]
        :param gameName: (string): The name of the game that will be displayed as title
        :param nameOfStrategy1: (string): The name that the first Strategy of player 1 displayed on x axis.
        :param nameOfStrategy2: (string): The name that the first Strategy of player 2 displayed on y axis.
        :param algorithm: (string): choose from:
                -"POGA": projected online gradient ascent, see Shalev-Swartz p.144
                -"EGA": entropic gradient ascent, see Shalev-Swartz p.144
        :param payoff1: (np.array): optional argument, only needed when stable points should be plotted
        :param payoff2: (np.array): same as payoff1
        :param stableTo: (list of lists): an optional argument. When not empty the region of stable point
                                            w.r.t to the given nash equilibria will be plotted
    """
    if stableTo is None:
        stableTo = []
    plt.figure(figsize=(6, 6), dpi=100)
    plt.scatter(p1Init, p2Init, color='#E37222', marker='o', s=70, zorder=5, label='initial strategy')
    plotEquilibria(pne, mne)
    plt.plot(p1, p2, marker='.', color='black')
    labelPlot(gameName, nameOfStrategy1, nameOfStrategy2, algorithm)
    legend(pne, mne)
    adjustPlot()
    plotStableRegions(payoff1, payoff2, stableTo)
    plt.show()


def computeStablePoints(payoff1, payoff2, ne):
    """ computes the variational stability (stability)
        for #granularity many points. Points correspond to initial strategies.
        The notion of stability is described in Mertikoloulos, Definition 2.3
        the function returns stable1 and stable2, the x,y coordinates of points that are stable.

    Args:
        :param payoff1: (np.array): payoff matrix player1
        :param payoff2: (np.array): payoff matrix player2
        :param ne: (np.array): Nash Equilibrium

    :return: stable1: (list): he x coordinates of points that are stable
    :return: stable2: (list): he y coordinates of points that are stable

    """
    # for zoomed in version
    # granularity = 1000
    granularity = 150
    p1 = p2 = np.linspace(0, 1, granularity)

    stable1 = []
    stable2 = []

    for x in p1:
        for y in p2:
            stability = checkStability(x, y, payoff1, payoff2, ne)
            if stability < 0:
                stable1 += [x]
                stable2 += [y]

    return stable1, stable2


def checkStability(x, y, payoff1, payoff2, ne):
    """ checks the variational Stability for one specific point
        w.r.t to the given Nash Equilibrium and returns the stability value

    Args:
    :param x: (float): x coordinate in [0,1]
    :param y: (float): y coordinate in [0,1]
    :param payoff1: (np.array): payoff matrix player1
    :param payoff2: (np.array): payoff matrix player2
    :param ne: (np.array): Nash Equilibrium
    :return: stability: (float): the result of the calculation
    """

    # the Nash Equilibrium as mixed extension from player 1's and player 2's perspective respectively
    ne1 = np.array([ne[0], 1 - ne[0]])
    ne2 = np.array([ne[1], 1 - ne[1]])

    strategy1 = np.array([x, 1 - x])
    strategy2 = np.array([y, 1 - y])

    grad1 = payoff1.dot(strategy2)
    grad2 = strategy1.transpose().dot(payoff2)

    stability1 = np.inner(grad1, strategy1 - ne1)
    stability2 = np.inner(grad2, strategy2 - ne2)

    stability = stability1 + stability2
    return stability


def plotStableRegion(stable1, stable2, color):
    """ plots the region that is variational stable with respect to some nash equilibrium

    Args:
        :param stable1: (list of float): the x coordinates of points that are stable
        :param stable2: (list of float): the x coordinates of points that are stable
        :param color: (string): the color of this region. Note that overlapping regions change in color.
    """
    coloring = [color] * len(stable1)
    plt.scatter(stable1, stable2, c=coloring, alpha=.1, zorder=0)


def plotStableRegions(payoff1, payoff2, stableTo):
    """ plots not all the stable regions given in the stableTo list
    Args:
    :param payoff1: (np.array): payoff matrix player1
    :param payoff2: (np.array): payoff matrix player2
    :param stableTo: (list of lists): list of nash equilibria
    """
    for x in range(len(stableTo)):
        stable1, stable2 = computeStablePoints(payoff1, payoff2, stableTo[x])
        plotStableRegion(stable1, stable2, colors[x])


def calculateFrequencies(p1Init, p2Init, iterations, payoff1, payoff2, algorithm):
    """ calculates the empirical frequency distribution given an algorithm

    :param p1Init: initial probability for Player 1 playing action 1
    :param p2Init: initial probability for Player 2 playing action 1
    :param iterations: number of iterations (time)
    :param payoff1: payoff matrix player 1
    :param payoff2: payoff matrix player 2
    :param algorithm: (string): choose from:
                -"POGA": projected online gradient ascent, see Shalev-Swartz p.144
                -"EGA": entropic gradient ascent, see Shalev-Swartz p.144

    :return: frequencyStrategy1Action1 (list): frequency of player 1 actually having played action 1 by following
                                                the algorithms update step
    :return: frequencyStrategy2Action1 (list): accordingly
    """

    numStrategy1Action1Played, numStrategy2Action1Played = 0, 0
    frequencyStrategy1Action1, frequencyStrategy2Action1 = [], []

    eta = calculateStepSize(iterations)

    strategy1 = np.array([p1Init, 1 - p1Init])
    strategy2 = np.array([p2Init, 1 - p2Init])

    for t in range(iterations):

        if strategy1[0] > 0.5:
            numStrategy1Action1Played += 1
        if strategy2[0] > 0.5:
            numStrategy2Action1Played += 1

        frequencyStrategy1Action1 += [numStrategy1Action1Played / (t + 1)]
        frequencyStrategy2Action1 += [numStrategy2Action1Played / (t + 1)]

        strategy1, strategy2 = updateStep(payoff1, payoff2, strategy1, strategy2, eta, algorithm)

    return frequencyStrategy1Action1, frequencyStrategy2Action1


def plotFrequencies(iterations, frequencyStrategy1Action1, frequencyStrategy2Action1, nameOfStrategy1Action1,
                    nameOfStrategy2Action1, gameName, algorithm):
    """
    :param iterations: time
    :param frequencyStrategy1Action1: output 1 of calculateFrequencies()
    :param frequencyStrategy2Action1: output 1 of calculateFrequencies()
    :param nameOfStrategy1Action1: name of Player 1's first action
    :param nameOfStrategy2Action1: name of Player 2's first action
    :param gameName: name of the Game
    :param algorithm: (string): choose from:
                -"POGA": projected online gradient ascent, see Shalev-Swartz p.144
                -"EGA": entropic gradient ascent, see Shalev-Swartz p.144
    """

    plt.figure(figsize=(6, 6), dpi=100)
    labelTitle(gameName, algorithm)

    # time on x axis
    x = range(iterations)

    plt.plot(x, frequencyStrategy1Action1, label='Player 1: ' + nameOfStrategy1Action1, c='#0065BD', ls='-')
    plt.plot(x, frequencyStrategy2Action1, label='Player 2: ' + nameOfStrategy2Action1, c='#A2AD00', ls='-')

    plt.ylim(-0.05, 1.05)
    plt.xlabel('time $t$', fontsize=10)
    plt.ylabel('frequency', fontsize=10)
    plt.legend()
    plt.show()


def calculateStepSize(iterations):
    """ for online mirror descent step size should be: eta = sqrt(log(d))/(L*sqrt(2T)) (Shalev-Shwartz, p.140)
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

    # eta = 0.3
    eta = 0.1
    return eta


def createMeshGrid(gridSize):
    """creates two meshgrids, i.e a grid of discrete probability pairs which size gridsize * gridsize

    Args:
        :param gridSize: (int):  number vor initial probability pairs. Number of vector will be gridSize^2

    Returns:
        (np.array): meshgrid for x axis (Player1)
        (np.array): meshgrid for y axis (Player2)
    """
    p1 = p2 = np.linspace(0, 1, gridSize)
    x, y = np.meshgrid(p1, p2)
    return x, y


def plotEquilibria(pne, mne):
    """subroutine for plotting the Equilibira using matplotlib

    Args:
        :param pne: (list of lists): array of all Pure Nash Equilibria. Syntax [[1,1],[0,0],...]
        :param mne: (list of lists): array of all Mixed Nash Equilibria.
            Syntax (..., (P1(strategy1),P2(strategy1)), ... ), e.g [[3/5,2/5]]
    """

    # PNE
    for k in range(len(pne)):
        i = pne[k][0]
        j = pne[k][1]
        # just to make sure to only print label once
        if k > 0:
            plt.scatter(i, j, color='#000000', marker='o', s=100, zorder=3)
        else:
            plt.scatter(i, j, color='#000000', marker='o', s=100, label='PNE', zorder=3)
    # MNE
    for k in range(len(mne)):
        i = mne[k][0]
        j = mne[k][1]
        # only label once
        if k > 0:
            plt.scatter(i, j, color='#000000', marker='*', s=150, zorder=3)
        else:
            plt.scatter(i, j, color='#000000', marker='*', s=150, label='MNE', zorder=3)


def labelPlot(gameName, nameOfStrategy1, nameOfStrategy2, algorithm):
    """give the plot a title and labels the axis accordingly

    Args:
        :param gameName: (string): The name of the game that will be displayed as title
        :param nameOfStrategy1: (string): The name that the first Strategy of player 1 displayed on x axis.
        :param nameOfStrategy2: (string): The name that the first Strategy of player 2 displayed on y axis.
        :param algorithm: (string): choose from:
                -"POGA": projected online gradient ascent, see Shalev-Swartz p.144
                -"EGA": entropic gradient ascent, see Shalev-Swartz p.144
    """

    # plt.xlabel('P(' + nameOfStrategy1 + ') Player 1', fontsize=15)
    # plt.ylabel('P(' + nameOfStrategy2 + ') Player 2', fontsize=15)
    plt.xlabel('$x_{1,' + nameOfStrategy1 + '}$', fontsize=15)
    plt.ylabel('$x_{2,' + nameOfStrategy2 + '}$', fontsize=15)
    labelTitle(gameName, algorithm)


def labelTitle(gameName, algorithm):
    """ Gives plot a title depending on the algorithm and the gamename provided in initialization

    """
    if algorithm == "POGA":
        algorithmName = "projected online gradient ascent"
    elif algorithm == "EGA":
        algorithmName = "entropic gradient ascent"
    else:
        algorithmName = ""
    title = gameName + ' \n ' + '(' + algorithmName + ')'
    plt.title(title, fontsize=14)


def legend(pne, mne):
    """ plots the Pure and Mixed Nash Equilibria if existing

    :param pne: Pure Nash Equilibria
    :param mne: Mixed Nash Equilibria
    """

    # comment in if you want a legend
    # if (len(pne) + len(mne)) > 0:
    #    plt.legend(fontsize=12, loc='best', bbox_to_anchor=(1, 1))


def adjustPlot():
    """makes small adjustments

    """
    plt.axis("scaled")
    plt.ylim(-0.05, 1.05)
    plt.xlim(-0.05, 1.05)

    # use to zoom in at [0,0]
    # plt.ylim(-0.01, .01)
    # plt.xlim(-0.01, .01)

    # use to zoom in at [2/5,3/5]
    # plt.ylim(2/5-0.1, 2/5+.1)
    # plt.xlim(3/5-0.1, 3/5+.1)

    # use to zoom in at [100/101,100/101]
    # plt.ylim(0.95, 1.0)
    # plt.xlim(0.95, 1.0)


def project(v):
    """ borrowed from Mathieu Blondel (https://gist.github.com/mblondel/6f3b7aaad90606b98f71)
        take a vector of any dimension and output the euclidean projection onto the simplex

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


def isFeasible(strategy):
    return sum(strategy) == 1 and min(strategy) >= 0 and max(strategy) <= 1

