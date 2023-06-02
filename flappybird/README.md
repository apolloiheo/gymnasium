# flappybird
## Objective
Use a simple genetic algorithm to create an agent that can play Flappy Bird perfectly. Neural network built from scratch with numpy.

## How to Run
With `Python 3.7.16` run `pip install -r requirements.txt.`
Run `python agent.py`.

## Future Improvements
1. bc of random worlds, to prevent overfitting on the worlds we see, we make each bird go thru each world X number of times n take some avg of the fitness of the birds
2. playing around suggests initial random weights matter a lot when seeing convergence:
2a. make world stop if say fitness more than triples, as we alr have a good contestant to avoid wasting time
2b. if fitness is not that much better than the rest, randomly generate new matrices
2c. let more than the top 2 reproduce (i was just lazy) lmao
do some hyperparameter tuning
3. feature engineering (ie like squared distances or smthg to give it an extra layer of nonlinearity

## References
