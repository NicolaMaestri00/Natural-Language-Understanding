''' This file contains the functions that are used in the training and evaluation loops '''

import math
import copy
import numpy as np
from tqdm import tqdm
import wandb
import torch
from torch import optim


def train_loop(data, optimizer, criterion, model, clip=5):
    ''' This function is used to train the model '''
    model.train()
    loss_array = []
    number_of_tokens = []

    for sample in data:
        optimizer.zero_grad()                                       # Zero the gradients
        output = model(sample['source'])                            # Forward pass
        loss = criterion(output, sample['target'])                  # Compute the loss
        loss_array.append(loss.item() * sample["number_tokens"])
        number_of_tokens.append(sample["number_tokens"])
        loss.backward()                                             # Backward pass
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)    # Clip the gradients
        optimizer.step()                                            # Update the weights

    ppl = math.exp(sum(loss_array) / sum(number_of_tokens))
    loss_to_return = sum(loss_array) / sum(number_of_tokens)

    return ppl, loss_to_return


def eval_loop(data, eval_criterion, model):
    ''' This function is used to evaluate the model '''
    model.eval()
    loss_to_return = []
    loss_array = []
    number_of_tokens = []

    with torch.no_grad():
        for sample in data:
            output = model(sample['source'])
            loss = eval_criterion(output, sample['target'])
            loss_array.append(loss.item())
            number_of_tokens.append(sample["number_tokens"])

    ppl = math.exp(sum(loss_array) / sum(number_of_tokens))
    loss_to_return = sum(loss_array) / sum(number_of_tokens)
    return ppl, loss_to_return


def train_loop_Nt_AvSGD(train_data, train_criterion, val_data, val_criterion, optimizer, model, clip=5):

    loss_array = []
    number_of_tokens = []

    # New parameters
    k = 1
    t = 0
    L = 25
    n = 5
    T = 0
    logs = []

    for sample in train_data:
        model.train()
        optimizer.zero_grad()                                       # Zero the gradients
        output = model(sample['source'])                            # Forward pass
        loss = train_criterion(output, sample['target'])            # Compute the loss
        loss_array.append(loss.item() * sample["number_tokens"])
        number_of_tokens.append(sample["number_tokens"])
        loss.backward()                                             # Backward pass
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)    # Clip the gradients
        optimizer.step(k, T)                                        # Update the weights

        if k % L == 0 and T == 0:
            # compute validation ppl
            val_ppl, _ = eval_loop(val_data, val_criterion, model)

            if t > n and val_ppl > min(logs[0:t-n]):
                T = k

            # append v to logs
            logs.append(val_ppl)
            t += 1

        k += 1

    ppl = math.exp(sum(loss_array) / sum(number_of_tokens))
    loss_to_return = sum(loss_array) / sum(number_of_tokens)

    return ppl, loss_to_return


def training(
        model,
        optimizer,
        n_epochs,
        init_patience,
        train_loader,
        criterion_train,
        clip,
        dev_loader,
        criterion_eval,
        nt_asgd_flag=False,
        log_wandb=False
        ):
    ''' This function is used to train the model '''

    patience = init_patience
    losses_train = []
    losses_dev = []
    sampled_epochs = []
    best_ppl = math.inf
    best_model = None

    # Training loop
    pbar = tqdm(range(1, n_epochs + 1))

    for epoch in pbar:
        # Train for one epoch and log training loss
        if nt_asgd_flag:
            ppl_train, loss_train = train_loop_Nt_AvSGD(train_loader,
                                                       criterion_train,
                                                       dev_loader,
                                                       criterion_eval,
                                                       optimizer,
                                                       model,
                                                       clip)
        else:
            ppl_train, loss_train = train_loop(train_loader,
                                               optimizer,
                                               criterion_train,
                                               model,
                                               clip)
        avg_train_loss = np.mean(loss_train)
        losses_train.append(avg_train_loss)

        if log_wandb:
            # Log training loss and perplexity to WandB
            wandb.log({ "Training_Loss": avg_train_loss,
                        "Training_Perplexity": ppl_train},
                        step=epoch)

        if epoch % 1 == 0:
            sampled_epochs.append(epoch)

            # Evaluate on validation set and log metrics every epoch
            ppl_dev, loss_dev = eval_loop(dev_loader, criterion_eval, model)
            avg_dev_loss = np.mean(loss_dev)
            losses_dev.append(avg_dev_loss)

            if log_wandb:
                # Log validation loss and perplexity to WandB
                wandb.log({ "Validation_Loss": avg_dev_loss,
                            "Validation_Perplexity": ppl_dev},
                            step=epoch)

            # Update progress bar with current perplexity
            pbar.set_description(f"PPL: {ppl_dev:.6f}")

            # Early stopping check
            if ppl_dev < best_ppl:
                best_ppl = ppl_dev
                best_model = copy.deepcopy(model).to('cpu')
                patience = init_patience
            else:
                patience -= 1

            if patience <= 0:
                print(f"Early stopping at epoch {epoch}")
                break

    return best_model


class Nt_AvSGD:
    def __init__(self, model, lr=1, momentum=0.9):
        """
        Non-Monotonically Triggered Averaged SGD optimizer.

        Args:
            model (nn.Module): The PyTorch model to optimize.
            lr (float): Learning rate for SGD.
            momentum (float): Momentum factor for SGD.
            threshold (int): Number of iterations without improvement to trigger averaging.
        """

        self.model = model
        self.base_optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)

        # Store current best loss and patience counter
        self.best_loss = float('inf')
        self.no_improve_count = 0

        # Initialize parameter average storage
        self.param_avg = {name: param.clone().detach() for name, param in model.named_parameters()}
        self.condition_already_met = False

    def step(self, k, T):
        """
        Performs a single optimization step with conditional averaging.

        Args:
            loss (float): The current training loss.
        """

        # Take a gradient descent step
        self.base_optimizer.step()

        if T != 0 and not self.condition_already_met:
            self.condition_already_met = True
            with torch.no_grad():
                for name, param in self.model.named_parameters():
                    self.param_avg[name].copy_(param)

        with torch.no_grad():
            for name, param in self.model.named_parameters():
                self.param_avg[name] = (1/(k-T)) * param + (k-T-1/(k-T)) * self.param_avg[name]

    def zero_grad(self):
        """Zero the gradients of all optimized parameters."""
        self.base_optimizer.zero_grad()
