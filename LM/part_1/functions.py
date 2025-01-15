''' This file contains the functions that are used in the training and evaluation loops '''

import math
import copy
import torch
import numpy as np
from tqdm import tqdm
import wandb


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
            output = model(sample['source'])                    # Forward pass
            loss = eval_criterion(output, sample['target'])     # Compute the loss
            loss_array.append(loss.item())
            number_of_tokens.append(sample["number_tokens"])

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
        ppl_train, loss_train = train_loop(train_loader, optimizer, criterion_train, model, clip)
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
