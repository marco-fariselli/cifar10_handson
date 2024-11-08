# Cifar10 hands-on

4 models have been trained on the Cifar10 dataset. Each model has some improvements wrt the previous one to make it faster on the GAP9 architecture.

## Exercise 1:

Complete the following table:


| Model | Float Accuracy | Quant Accuracy | Ops    | Parameters | Coeff Size deployment* | Cyc    | Op/Cyc | Why is this better than previous? |
|-------|----------------|----------------|--------|------------|------------------------|--------|--------|-----------------------------------|
| v1    |                |                |        |            |                        |        |        |                                   |
| v2    |                |                |        |            |                        |        |        |                                   |
| v3    |                |                |        |            |                        |        |        |                                   |
| v4    |                |                |        |            |                        |        |        |                                   |

*Size of the NN coefficients to deploy (flash usage).

## Exercise 2:

Apply optimizations to the deployment scripts and make the previous table faster.

## Exercise 3:

Make the best possible model (the fastest / energy efficient) with an accuracy of at least 50% on the cifar small (the one used in the deployment script) testset. You can train from scratch a new model, or play with NNTool optimizations.

Best model wins a prize :)
