# Assignment 1: *Languange models and regularization strategies*

## 1. Introduction

In this assignment, we develop a *language model* (LM) to represent the probability distribution over a sequence of tokens. Starting with a *vanilla recurrent neural network* (RNN) followed by a linear and softmax layer trained using *stochastic gradient descent*, we explore various enhancements. Experiments incorporating *Long-Short Term Memory* (LSTM) networks, *dropout layers*, and the *AdamW* optimizer result in a significant reduction in test set perplexity. Applying *Weight Tying*, *Variational Dropout*, and *Non-monotonically Triggered Averaged SGD* (NT-ASGD) further reduce the perplexity to $104$, surpassing the baseline by more than $50$ points.

## 2. Implementation details

<!-- Experiment 1: Baseline -->
In the first experiment, we use a *vanilla RNN* followed by a linear and softmax layer. Enhancements include replacing the *RNN* with an *LSTM*, applying *dropout* to both embedding and hidden layers, and utilizing the *AdamW* optimizer. Model hyperparameters are provided in Table [LSTM Hyperparameters](#lstm-hyperparameters).

<!-- Experiment 2: Baseline -->
In the second experiment, we implement *Weight Tying*, *Variational Dropout* and *Non-monotonically Triggered Averaged SGD*, three regularization techniques outlined in [Merity et al. (2017)](https://arxiv.org/abs/1708.02182).

*Weight Tying* shares weights for projecting the vocabulary into the embedding space and projecting the hidden state into the vocabulary. An additional linear layer aligns the hidden state with the embedding space dimensions.

*Variational Dropout* applies a unique binary mask for all sequences in the same batch. The mask is sampled according to a *dropout probability* \\(p\\) and scaled by \\(\\frac{1}{1-p}\\) to maintain the expected value of the activations.

*Non-monotonically Triggered Averaged SGD* is implemented following the pseudocode in Figure [Pseudocode](#non-monotonically-triggered-asgd-algorithm). During each epoch, we perform steps with *SGD* while maintaining an updated average of the weights starting from a dynamically determined iteration \\(T\\). At the end of each epoch, averaged weights are used to update the model, and hyperparameters are reinitialized before starting the next epoch. Momentum is set to \\(0.9\\) in *SGD*, improving optimization by smoothing updates.

<!-- Training -->
For both experiments, training runs for a maximum of \\(50\\) epochs, with early stopping triggered after \\(5\\) consecutive validation stagnations. Hyperparameter tuning is conducted to optimize the learning rates for both *SGD* and *AdamW* optimizers. To ensure stability, gradients are clipped to a norm of \\(5\\). Table [Training Hyperparameters](#training-hyperparameters) provides the full training configuration.

<!-- Loss Function -->
The model is trained using the *Cross Entropy* loss function, excluding *\<pad\>* tokens, while *Perplexity* is considered for evaluation.

<!-- Dataset -->
We use the *PennTreeBank* dataset for training and evaluation. The vocabulary is derived from the corpus with two additional tokens: *\<eos\>* to mark the end of a sequence and *\<pad\>* to ensure uniform sequence lengths. Parameters for the data loaders are listed in Table [Loader Hyperparameters](#loader-hyperparameters).


## 3. Results

In all experiments, we report the average score over \(5\) runs along with the relative standard deviation. Results of both experiments are listed in [Table Final PPL](#table-final-ppl), and the best scores are visualized in [Figure Final PPL](#figure-final-ppl).

The *Vanilla RNN*-based model trained with *SGD* achieves a test perplexity of \(158.42\) when the *learning rate* is set to \(1\). This score serves as a reference point for subsequent experiments.

Replacing the *Vanilla RNN* with an *LSTM* reduces the test perplexity to \(142.11\). Incorporating *dropout layers* significantly reduces test perplexity to \(123.76\). These results are obtained using *SGD* with a learning rate of \(2\). Replacing *SGD* with *AdamW* and setting the learning rate to \(5 \times 10^{-4}\) further improves the final perplexity, achieving a score of \(121.43\). [Figures ex1 Train Loss](#figure-ex1-train-loss) and [ex1 Val Loss](#figure-ex1-val-loss) illustrate the progression of training and validation losses for the models mentioned above.

Substituting *Dropout* with *Variational Dropout* does not improve the performance, with a score of \(121.40\), while the addition of *Weight Tying* results in a test perplexity of \(107.76\). Using the *NT-ASGD* optimizer to train an *LSTM* model with *dropout layers* leads to a test perplexity of \(130.04\). Nevertheless, combining an *LSTM* model with *variational dropout* and the *NT-ASGD* optimizer achieves a perplexity of \(104.27\) on the test set, which is the best score obtained. [Figures ex2 Train Loss](#figure-ex2-train-loss) and [ex2 Val Loss](#figure-ex2-val-loss) show the progression of training and validation losses for the aforementioned models. It is interesting to observe that models trained with the *NT-ASGD* optimizer exhibit a validation loss after just one epoch that is half of that achieved by models trained with *AdamW*. Furthermore, training with *NT-ASGD* converges in fewer than \(15\) epochs, demonstrating that this optimizer enables faster learning in this scenario.
