# Henning Heyen Bachelor Project

This project addresses the outcome of no-regret dynamics in finite games. Can the outcome
be characterized by traditional game-theoretic solution concepts like Nash equilibria?
The general answer to this question is no. Nevertheless, there are some games where
Nash convergence under no-regret learning has been observed before. 

The project aims to give a neat and compact overview on sufficient conditions under which no-regret learning
converges to Nash. These conditions are empirically confirmed by employing two concrete
instances of no-regret algorithms on simple two-player games (*Projected Online Gradient Ascent* and *Entropic Gradient Ascent*). Plots are provided further
to give an intuition of the algorithms’ behavior. 

The bottom line is that only strict Nash equilibria survive under no-regret dynamics. Moreover, the empirical frequency of play
converges to interior Nash equilibria in two-player zero-sum games.

Feel free to play around with the code and to add new games (2x2 or 3x3)!

## No-Regret Learning in Finite Games

The Repository has the following folders: 

* **Figures**
  * All plots and figures that were used in the Bachelor Thesis

* **OnlineMirrorAscent**
  * Complete Python to run the experiments
  * Most important files: 
    1. **TwoTimesTwoGamesFunctions**: code for 2x2 games. Contains both *Projected Online Gradient Ascent* as well as *Entropic Gradient Ascent* algorithms
    2. **ThreeTimesThreeFunctions**: Analog code for 3x3 games
    3. The other files are Jupyter Notebooks where the algorithms are run and visualised on simple 2-Player games
    4. **PayoffMatrices**: Folder that contains images (.jpeg or .png) of the payoff matrices imported at the beginning of each Jupyter notebook 

* **Presentation**
  * Bachelor Colloquium slides in pdf as well as powerpoint version

* **Thesis** 
  * Contains the Bachelor Thesis in PDF version as well as the *Overleaf* Latex Repository

