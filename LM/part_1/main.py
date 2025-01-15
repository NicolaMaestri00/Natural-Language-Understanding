''' This file is used to run the functions and print the results  '''

from functools import partial
import wandb

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np

from utils import read_file, Lang, PennTreeBank, collate_fn
from model import LmRnn, LmLstm, init_weights
from functions import eval_loop, training


if __name__ == "__main__":

    # Log results to WandB
    LOG_WANDB = False
    if LOG_WANDB:
        PROJECT_NAME = "NLU Assignment 1"

    # Set the device
    DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    # Load the data
    try:
        # Read the files
        train_raw = read_file("239920_nicola_maestri/LM/part_1/dataset/PennTreeBank/ptb.train.txt")
        dev_raw = read_file("239920_nicola_maestri/LM/part_1/dataset/PennTreeBank/ptb.valid.txt")
        test_raw = read_file("239920_nicola_maestri/LM/part_1/dataset/PennTreeBank/ptb.test.txt")
    except FileNotFoundError:
        try:
            # Try alternative paths
            train_raw = read_file("LM/part_1/dataset/PennTreeBank/ptb.train.txt")
            dev_raw = read_file("LM/part_1/dataset/PennTreeBank/ptb.valid.txt")
            test_raw = read_file("LM/part_1/dataset/PennTreeBank/ptb.test.txt")
        except FileNotFoundError as e:
            raise FileNotFoundError("The dataset files could not be found in the specified paths!\n\
                                    Please make sure the paths are correct and try again.") from e

    # Create the vocabulary
    vocabulary = Lang(train_raw, ["<pad>", "<eos>"])

    # Create the dataset
    train_dataset = PennTreeBank(train_raw, vocabulary)
    dev_dataset = PennTreeBank(dev_raw, vocabulary)
    test_dataset = PennTreeBank(test_raw, vocabulary)

    # Data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=64,
        collate_fn=partial(collate_fn, pad_token=vocabulary.word2id["<pad>"]),
        shuffle=True
        )
    dev_loader = DataLoader(
        dev_dataset,
        batch_size=128,
        collate_fn=partial(collate_fn, pad_token=vocabulary.word2id["<pad>"])
        )
    test_loader = DataLoader(
        test_dataset,
        batch_size=128,
        collate_fn=partial(collate_fn, pad_token=vocabulary.word2id["<pad>"])
        )

    # Define model parameters
    HID_SIZE = 200
    EMB_SIZE = 300
    VOCABULARY_LENGTH = len(vocabulary.word2id)

    # Define hyperparameters
    N_RUNS = 1
    N_EPOCHS = 50
    PATIENCE_INITIAL = 5
    CLIP = 5

    # Loss function
    criterion_train = nn.CrossEntropyLoss(ignore_index=vocabulary.word2id["<pad>"])
    criterion_eval = nn.CrossEntropyLoss(ignore_index=vocabulary.word2id["<pad>"], reduction='sum')


    # ---------------------------------------------------------
    # Ex 1.0: Vanilla RNN + SGD [ Best learning rate ]
    # ---------------------------------------------------------

    learning_rates = [0.1, 1, 2]

    # Loop over each learning rate
    for lr in learning_rates:

        # Average over multiple runs
        final_PPLs = []
        flag = True
        for _ in range(N_RUNS):

            # Initialize the model
            model = LmRnn(  EMB_SIZE,
                            HID_SIZE,
                            VOCABULARY_LENGTH,
                            pad_index=vocabulary.word2id["<pad>"]
                            ).to(DEVICE)

            model.apply(init_weights)

            # Optimizer
            optimizer = optim.SGD(model.parameters(), lr=lr)

            if LOG_WANDB and flag:
                # Initialize a new WandB run for each learning rate
                MODEL_NAME = "Vanilla_RNN_SGD_lr_" + str(lr)
                wandb.init(project=PROJECT_NAME, name=MODEL_NAME, reinit=True)

                # Log hyperparameters to WandB
                wandb.config.update({
                    "model": MODEL_NAME,
                })


            # Training
            best_model = training(  model,
                                    optimizer,
                                    N_EPOCHS,
                                    PATIENCE_INITIAL,
                                    train_loader,
                                    criterion_train,
                                    CLIP,
                                    dev_loader,
                                    criterion_eval,
                                    LOG_WANDB and flag
                                    )
            if LOG_WANDB and flag:
                wandb.finish()

            # Load the best model and evaluate on the test set
            best_model.to(DEVICE)
            final_ppl, _ = eval_loop(test_loader, criterion_eval, best_model)

            # # Save the model
            # if flag:
            #     torch.save(best_model.state_dict(), "239920_nicola_maestri/LM/part_1/bin/Vanilla_RNN_SGD_lr_" + str(lr) + ".pt")

            final_PPLs.append(final_ppl)
            flag = False

        # Average the final perplexities over multiple runs
        final_PPLs = np.asarray(final_PPLs)
        avg_final_ppl = round(final_PPLs.mean(), 3)
        std_final_ppl = round(final_PPLs.std(), 3)

        # Print the average perplexity and standard deviation
        print("Average test PPL for Vanilla_RNN_SGD_lr_" + str(lr) + ": ", avg_final_ppl, "±", std_final_ppl)

        if LOG_WANDB:
            # Initialize a new WandB run for the average results
            MODEL_NAME = "AVG_Vanilla_RNN_SGD_lr_" + str(lr)
            wandb.init(project=PROJECT_NAME, name=MODEL_NAME, reinit=True)
            # Log hyperparameters to WandB
            wandb.config.update({
                "model": MODEL_NAME,
            })
            wandb.log({"Test_Perplexity": avg_final_ppl,
                    "Test_Perplexity Std": std_final_ppl})
            # Finish the current WandB run
            wandb.finish()


    # ---------------------------------------------------------
    # Ex 1.1: LSTM + SGD
    # ---------------------------------------------------------

    learning_rates = [0.5, 1, 2]

    # Loop over each learning rate
    for lr in learning_rates:

        # Average over multiple runs
        final_PPLs = []
        flag = True

        for _ in range(N_RUNS):
            # Initialize the model
            model = LmLstm( EMB_SIZE,
                        HID_SIZE,
                        VOCABULARY_LENGTH,
                        pad_index=vocabulary.word2id["<pad>"]
                        ).to(DEVICE)

            model.apply(init_weights)

            # Optimizer
            optimizer = optim.SGD(model.parameters(), lr=lr)

            if LOG_WANDB and flag:
                # Initialize a new WandB run for each learning rate
                MODEL_NAME = "LSTM_SGD_lr_" + str(lr)
                wandb.init(project=PROJECT_NAME, name=MODEL_NAME, reinit=True)

                # Log hyperparameters to WandB
                wandb.config.update({
                    "model": MODEL_NAME,
                })

            # Training
            trained_model = training(  model,
                                    optimizer,
                                    N_EPOCHS,
                                    PATIENCE_INITIAL,
                                    train_loader,
                                    criterion_train,
                                    CLIP,
                                    dev_loader,
                                    criterion_eval,
                                    LOG_WANDB and flag
                                    )
            
            if LOG_WANDB and flag:
                wandb.finish()

            # Load the best model and evaluate on the test set
            trained_model.to(DEVICE)
            final_ppl, _ = eval_loop(test_loader, criterion_eval, trained_model)

            # # Save the model
            # if flag:
            #     torch.save(trained_model.state_dict(), "239920_nicola_maestri/LM/part_1/bin/LSTM_SGD_lr_" + str(lr) + ".pt")

            final_PPLs.append(final_ppl)
            flag = False

        # Average the final perplexities over multiple runs
        final_PPLs = np.asarray(final_PPLs)
        avg_final_ppl = round(final_PPLs.mean(), 3)
        std_final_ppl = round(final_PPLs.std(), 3)

        # Print the average perplexity and standard deviation
        print("Average test PPL for LSTM_SGD_lr_" + str(lr) + ": ", avg_final_ppl, "±", std_final_ppl)

        if LOG_WANDB:
            # Initialize a new WandB run for the average results
            MODEL_NAME = "AVG_LSTM_SGD_lr_" + str(lr)
            wandb.init(project=PROJECT_NAME, name=MODEL_NAME, reinit=True)
            # Log hyperparameters to WandB
            wandb.config.update({
                "model": MODEL_NAME,
            })
            wandb.log({"Test_Perplexity": avg_final_ppl,
                    "Test_Perplexity Std": std_final_ppl})
            # Finish the current WandB run
            wandb.finish()


    # ---------------------------------------------------------
    # Ex 1.2: LSTM + SGD + Do
    # ---------------------------------------------------------

    learning_rates = [0.5, 1, 2]

    # Loop over each learning rate
    for lr in learning_rates:
        # Average over multiple runs
        final_PPLs = []
        flag = True

        for _ in range(N_RUNS):

            # Initialize the model
            model = LmLstm( EMB_SIZE,
                            HID_SIZE,
                            VOCABULARY_LENGTH,
                            pad_index=vocabulary.word2id["<pad>"],
                            dropout=True
                            ).to(DEVICE)

            model.apply(init_weights)

            # Optimizer
            optimizer = optim.SGD(model.parameters(), lr=lr)

            if LOG_WANDB and flag:
                # Initialize a new WandB run for each learning rate
                MODEL_NAME = "LSTM_SGD_Do_lr_" + str(lr)
                wandb.init(project=PROJECT_NAME, name=MODEL_NAME, reinit=True)

                # Log hyperparameters to WandB
                wandb.config.update({
                    "model": MODEL_NAME,
                })


            # Training
            trained_model = training(  model,
                                    optimizer,
                                    N_EPOCHS,
                                    PATIENCE_INITIAL,
                                    train_loader,
                                    criterion_train,
                                    CLIP,
                                    dev_loader,
                                    criterion_eval,
                                    LOG_WANDB and flag
                                    )
            
            if LOG_WANDB and flag:
                wandb.finish()

            # Load the best model and evaluate on the test set
            trained_model.to(DEVICE)
            final_ppl, _ = eval_loop(test_loader, criterion_eval, trained_model)

            # # Save the model
            # if flag:
            #     torch.save(trained_model.state_dict(), "239920_nicola_maestri/LM/part_1/bin/LSTM_SGD_Do_lr_" + str(lr) + ".pt")

            final_PPLs.append(final_ppl)
            flag = False

        # Average the final perplexities over multiple runs
        final_PPLs = np.asarray(final_PPLs)
        avg_final_ppl = round(final_PPLs.mean(), 3)
        std_final_ppl = round(final_PPLs.std(), 3)

        # Print the average perplexity and standard deviation
        print("Average test PPL for LSTM_SGD_Do_lr_" + str(lr) + ": ", avg_final_ppl, "±", std_final_ppl)

        if LOG_WANDB:
            # Initialize a new WandB run for the average results
            MODEL_NAME = "AVG_LSTM_SGD_Do_lr_" + str(lr)
            wandb.init(project=PROJECT_NAME, name=MODEL_NAME, reinit=True)
            # Log hyperparameters to WandB
            wandb.config.update({
                "model": MODEL_NAME,
            })
            wandb.log({"Test_Perplexity": avg_final_ppl,
                    "Test_Perplexity Std": std_final_ppl})
            # Finish the current WandB run
            wandb.finish()


    # ---------------------------------------------------------
    # Ex 1.3: LSTM + AdamW + Do
    # ---------------------------------------------------------

    learning_rates = [0.0001, 0.0005, 0.001]

    # Loop over each learning rate
    for lr in learning_rates:
        # Average over multiple runs
        final_PPLs = []
        flag = True

        for _ in range(N_RUNS):

            # Initialize the model
            model = LmLstm( EMB_SIZE,
                            HID_SIZE,
                            VOCABULARY_LENGTH,
                            pad_index=vocabulary.word2id["<pad>"],
                            dropout=True
                            ).to(DEVICE)

            model.apply(init_weights)

            # Optimizer
            optimizer = optim.AdamW(model.parameters(), lr=lr)

            if LOG_WANDB and flag:
                # Initialize a new WandB run for each learning rate
                MODEL_NAME = "LSTM_AdamW_Do_lr_" + str(lr)
                wandb.init(project=PROJECT_NAME, name=MODEL_NAME, reinit=True)

                # Log hyperparameters to WandB
                wandb.config.update({
                    "model": MODEL_NAME,
                })

            # Training
            trained_model = training(  model,
                                    optimizer,
                                    N_EPOCHS,
                                    PATIENCE_INITIAL,
                                    train_loader,
                                    criterion_train,
                                    CLIP,
                                    dev_loader,
                                    criterion_eval,
                                    LOG_WANDB and flag
                                    )
            
            if LOG_WANDB and flag:
                wandb.finish()

            # Load the best model and evaluate on the test set
            trained_model.to(DEVICE)
            final_ppl, _ = eval_loop(test_loader, criterion_eval, trained_model)

            # # Save the model
            # if flag:
            #     torch.save(trained_model.state_dict(), "239920_nicola_maestri/LM/part_1/bin/LSTM_AdamW_Do_lr_" + str(lr) + ".pt")

            final_PPLs.append(final_ppl)
            flag = False

        # Average the final perplexities over multiple runs
        final_PPLs = np.asarray(final_PPLs)
        avg_final_ppl = round(final_PPLs.mean(), 3)
        std_final_ppl = round(final_PPLs.std(), 3)

        # Print the average perplexity and standard deviation
        print("Average test PPL for LSTM_AdamW_Do_lr_" + str(lr) + ": ", avg_final_ppl, "±", std_final_ppl)

        if LOG_WANDB:
            # Initialize a new WandB run for the average results
            MODEL_NAME = "AVG_LSTM_AdamW_Do_lr_" + str(lr)
            wandb.init(project=PROJECT_NAME, name=MODEL_NAME, reinit=True)
            # Log hyperparameters to WandB
            wandb.config.update({
                "model": MODEL_NAME,
            })
            wandb.log({"Test_Perplexity": avg_final_ppl,
                    "Test_Perplexity Std": std_final_ppl})
            # Finish the current WandB run
            wandb.finish()
