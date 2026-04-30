from util import manhattanDistance
from game import Directions
import random
import util
from typing import Any, DefaultDict, List, Set, Tuple

from game import Agent
from pacman import GameState


class ReflexAgent(Agent):
    """
      A reflex agent chooses an action at each choice point by examining
      its alternatives via a state evaluation function.

      The code below is provided as a guide.  You are welcome to change
      it in any way you see fit, so long as you don't touch our method
      headers.
    """

    def __init__(self):
        self.lastPositions = []
        self.dc = None

    def getAction(self, gameState: GameState):
        """
        getAction chooses among the best options according to the evaluation function.

        getAction takes a GameState and returns some Directions.X for some X in the set {North, South, West, East}
        ------------------------------------------------------------------------------
        Description of GameState and helper functions:

        A GameState specifies the full game state, including the food, capsules,
        agent configurations and score changes. In this function, the |gameState| argument
        is an object of GameState class. Following are a few of the helper methods that you
        can use to query a GameState object to gather information about the present state
        of Pac-Man, the ghosts and the maze.

        gameState.getLegalActions(agentIndex):
            Returns the legal actions for the agent specified. Returns Pac-Man's legal moves by default.

        gameState.generateSuccessor(agentIndex, action):
            Returns the successor state after the specified agent takes the action.
            Pac-Man is always agent 0.

        gameState.getPacmanState():
            Returns an AgentState object for pacman (in game.py)
            state.configuration.pos gives the current position
            state.direction gives the travel vector

        gameState.getGhostStates():
            Returns list of AgentState objects for the ghosts

        gameState.getNumAgents():
            Returns the total number of agents in the game

        gameState.getScore():
            Returns the score corresponding to the current state of the game


        The GameState class is defined in pacman.py and you might want to look into that for
        other helper methods, though you don't need to.
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(
            gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(
            len(scores)) if scores[index] == bestScore]
        # Pick randomly among the best
        chosenIndex = random.choice(bestIndices)

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState: GameState, action: str) -> float:
        """
        The evaluation function takes in the current GameState (defined in pacman.py)
        and a proposed action and returns a rough estimate of the resulting successor
        GameState's value.

        The code below extracts some useful information from the state, like the
        remaining food (oldFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        oldFood = currentGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [
            ghostState.scaredTimer for ghostState in newGhostStates]

        return successorGameState.getScore()


def scoreEvaluationFunction(currentGameState: GameState):
    """
      This default evaluation function just returns the score of the state.
      The score is the same one displayed in the Pacman GUI.

      This evaluation function is meant for use with adversarial search agents
      (not reflex agents).
    """
    min_ghostDistance = float('inf')
    currentScore = currentGameState.getScore()

    for ghost in currentGameState.getGhostPositions():
        currentDistance = manhattanDistance(ghost, currentGameState.getPacmanPosition())
        if currentDistance < min_ghostDistance: min_ghostDistance = currentDistance

    return (currentScore - 100 * (1.0 / min_ghostDistance))
    """
    return currentGameState.getScore()
    """


class MultiAgentSearchAgent(Agent):
    """
      This class provides some common elements to all of your
      multi-agent searchers.  Any methods defined here will be available
      to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

      You *do not* need to make any changes here, but you can if you want to
      add functionality to all your adversarial search agents.  Please do not
      remove anything, however.

      Note: this is an abstract class: one that should not be instantiated.  It's
      only partially specified, and designed to be extended.  Agent (game.py)
      is another abstract class.
    """

    def __init__(self, evalFn='scoreEvaluationFunction', depth='2'):
        self.index = 0  # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)


######################################################################################
# Problem 1b: implementing minimax


class MinimaxAgent(MultiAgentSearchAgent):
    """
      Your minimax agent (problem 1)
    """

    def __init__(self, evalFn='scoreEvaluationFunction', depth='2'):
        super().__init__(evalFn, depth)
        self._numMovimientos = 0

    def getAction(self, gameState: GameState) -> str:
        # BEGIN_YOUR_CODE (our solution is 22 lines of code, but don't worry if you deviate from this)
        numAgentes = gameState.getNumAgents()

        def minimax(estado, agente, profundidad):
            if estado.isWin() or estado.isLose():
                return estado.getScore()  # si hemos ganado devolvemos el resultado
            if profundidad == 0:
                return self.evaluationFunction(estado)  # si la profundidad es 0 no tenemos mas que hacer

            accionesLegales = estado.getLegalActions(agente)
            siguienteAgente = (agente + 1) % numAgentes  # aqui calculamos si el siguiente nodo es MAX O MIN
            # al ser 2: 0(MAX) 1(MIN), si podemos cambiar esta forma de calcularlo la cambiamos por algo mas bonito
            siguienteProfundidad = profundidad - 1 if siguienteAgente == 0 else profundidad  # aqui reducimos la profundidad segun la ronda            if agente == 0: # que sea un nodo MAX
            if agente == 0:  # nodo MAX
                valor = float('-inf')
                for accion in accionesLegales:
                    sucesor = estado.generateSuccessor(agente, accion)
                    valor = max(valor, minimax(sucesor, siguienteAgente, siguienteProfundidad))
                return valor
            else:  # nodo MIN
                valor = float('inf')
                for accion in accionesLegales:
                    sucesor = estado.generateSuccessor(agente, accion)
                    valor = min(valor, minimax(sucesor, siguienteAgente, siguienteProfundidad))
                return valor

        accionesLegales = gameState.getLegalActions(self.index)
        mejorAccion = None
        mejorValor = float('-inf')
        self._numMovimientos += 1

        for accion in accionesLegales:
            sucesor = gameState.generateSuccessor(self.index, accion)
            valor = minimax(sucesor, (self.index + 1) % numAgentes, self.depth)
            if valor > mejorValor:
                mejorAccion = accion
                mejorValor = valor
        return mejorAccion

    def final(self, state):
        print(f"Movimientos totales: {self._numMovimientos}")


######################################################################################
# Problem 2a: implementing alpha-beta


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (problem 2)
      You may reference the pseudocode for Alpha-Beta pruning here:
      en.wikipedia.org/wiki/Alpha%E2%80%93beta_pruning#Pseudocode
    """

    def __init__(self, evalFn='scoreEvaluationFunction', depth='2'):
        super().__init__(evalFn, depth)
        self._numMovimientos = 0

    def getAction(self, gameState: GameState) -> str:
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """

        # BEGIN_YOUR_CODE (our solution is 43 lines of code, but don't worry if you deviate from this)
        numAgentes = gameState.getNumAgents()

        def alphabeta(estado, agente, profundidad, alfa, beta):
            if estado.isWin() or estado.isLose():
                return estado.getScore()  # si hemos ganado devolvemos el resultado
            if profundidad == 0:
                return self.evaluationFunction(estado)  # si la profundidad es 0 no tenemos mas que hacer

            accionesLegales = estado.getLegalActions(agente)
            siguienteAgente = (agente + 1) % numAgentes  # aqui calculamos si el siguiente nodo es MAX O MIN
            siguienteProfundidad = profundidad - 1 if siguienteAgente == 0 else profundidad  # aqui reducimos la profundidad segun la ronda

            if agente == 0:  # nodo MAX
                valor = float('-inf')
                for accion in accionesLegales:
                    sucesor = estado.generateSuccessor(agente, accion)
                    valor = max(valor, alphabeta(sucesor, siguienteAgente, siguienteProfundidad, alfa, beta))

                    # hacemos la poda antes de devolver el valor
                    if valor > beta:
                        return valor
                    alfa = max(alfa, valor)
                return valor

            else:  # nodo MIN
                valor = float('inf')
                for accion in accionesLegales:
                    sucesor = estado.generateSuccessor(agente, accion)
                    valor = min(valor, alphabeta(sucesor, siguienteAgente, siguienteProfundidad, alfa, beta))

                    # hacemos la poda antes de devolver el valor
                    if valor < alfa:
                        return valor
                    beta = min(beta, valor)
                return valor

        accionesLegales = gameState.getLegalActions(self.index)
        mejorAccion = None
        mejorValor = float('-inf')

        alfa = float('-inf')
        beta = float('inf')

        self._numMovimientos += 1

        for accion in accionesLegales:
            sucesor = gameState.generateSuccessor(self.index, accion)
            valor = alphabeta(sucesor, (self.index + 1) % numAgentes, self.depth, alfa, beta)
            if valor > mejorValor:
                mejorAccion = accion
                mejorValor = valor

            alfa = max(alfa, mejorValor)

        return mejorAccion

    def final(self, state):
        print(f"Movimientos totales: {self._numMovimientos}")


######################################################################################
# Problem 3b: implementing expectimax


class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (problem 3)
    """

    def __init__(self, evalFn='scoreEvaluationFunction', depth='2'):
        super().__init__(evalFn, depth)
        self._numMovimientos = 0

    def getAction(self, gameState: GameState) -> str:
        """
          Returns the expectimax action using self.depth and self.evaluationFunction

          All ghosts should be modeled as choosing uniformly at random from their
          legal moves.
        """
        # BEGIN_YOUR_CODE
        numAgentes = gameState.getNumAgents()

        def expectimax(estado, agente, profundidad):
            if estado.isWin() or estado.isLose():
                return estado.getScore()
            if profundidad == 0:
                return self.evaluationFunction(estado)

            accionesLegales = estado.getLegalActions(agente)
            siguienteAgente = (agente + 1) % numAgentes
            siguienteProfundidad = profundidad - 1 if siguienteAgente == 0 else profundidad

            if agente == 0:  # nodo max
                valor = float('-inf')
                for accion in accionesLegales:
                    sucesor = estado.generateSuccessor(agente, accion)
                    valor = max(valor, expectimax(sucesor, siguienteAgente, siguienteProfundidad))
                return valor

            else:  # nodo aleatorio
                valorEsperado = 0
                for accion in accionesLegales:
                    sucesor = estado.generateSuccessor(agente, accion)
                    valorEsperado += expectimax(sucesor, siguienteAgente, siguienteProfundidad)

                return valorEsperado / len(accionesLegales)  # devolvemos la media

        accionesLegales = gameState.getLegalActions(self.index)
        mejorAccion = None
        mejorValor = float('-inf')
        self._numMovimientos += 1

        for accion in accionesLegales:
            sucesor = gameState.generateSuccessor(self.index, accion)
            valor = expectimax(sucesor, (self.index + 1) % numAgentes, self.depth)

            if valor > mejorValor:
                mejorAccion = accion
                mejorValor = valor
        return mejorAccion

    def final(self, state):
        print(f"Movimientos totales: {self._numMovimientos}")


######################################################################################
# Problem 4a (extra credit): creating a better evaluation function


def betterEvaluationFunction(currentGameState: GameState) -> float:
    """
      Your extreme, unstoppable evaluation function (problem 4). Note that you can't fix a seed in this function.
    """

    # BEGIN_YOUR_CODE (our solution is 16 lines of code, but don't worry if you deviate from this)
    raise Exception("Not implemented yet")
    # END_YOUR_CODE


# Abbreviation
better = betterEvaluationFunction
