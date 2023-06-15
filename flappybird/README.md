# flappybird
Timestamp: June 1, 2023 -- June 15, 2023

## Objective
Use a simple genetic algorithm to create an agent that can play Flappy Bird perfectly. Neural network built from scratch with numpy.

## How to Run
With `Python 3.7.16` run `pip install -r requirements.txt.`
Run `python agent.py`.

## Current Model Parameters
* **feature_map:** A function that takes in the two observations, horizontal and vertical distance to next pipe, and outputs features to be used as inputs to the neural network. (Identity)
* **hidden_dim:** Hidden layer size. (6)
* **mutation_scale** Mutations occur on each child via a zero-mean normal distribution. This sets its standard deviation. (1e-2)
* **population_size:** Population size for each generations beyond first. (10)
* **generations:** How many generations to run this simulation for. (10)
* **reward:** How to determine fitness of each bird. With this implementation, only the top 2 performers reproduce. (+ 0.1 per frame alive + |1-|vertical distance to next pipe||)
* **initial_pop:** First generation population size filled with randomized parameters. Default to population size, but having a big initial population helps there exist at least a couple successful birds to reproduce.

## Future Improvements
1. bc of random worlds, to prevent overfitting on the worlds we see, we make each bird go thru each world X number of times n take some avg of the fitness of the birds
2. playing around suggests initial random weights matter a lot when seeing convergence:
2a. make world stop if say fitness more than triples, as we alr have a good contestant to avoid wasting time
2b. if fitness is not that much better than the rest, randomly generate new matrices
2c. let more than the top 2 reproduce (i was just lazy) lmao
do some hyperparameter tuning
3. feature engineering (ie like squared distances or smthg to give it an extra layer of nonlinearity)

