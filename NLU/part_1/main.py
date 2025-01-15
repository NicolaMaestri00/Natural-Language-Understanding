''' This file is used to run the functions and print the results  '''


import os
import torch
import torch.nn as nn
from tqdm import tqdm
import wandb
from torch.utils.data import DataLoader
import torch.optim as optim
import numpy as np

from functions import training, eval_loop
from model import ModelIAS, init_weights
from utils import get_data, Lang, IntentsAndSlots, collate_fn


if __name__ == "__main__":

    # Log results to WandB
    LOG_WANDB = False
    if LOG_WANDB:
        PROJECT_NAME = "NLU Assignment 2 Nicola Maestri 239920"

    # Set the device
    DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
    PAD_TOKEN = 0

    # Load the data
    TRAIN_DATA_PATH = "NLU/part_1/dataset/ATIS/train.json"
    TEST_DATA_PATH = "NLU/part_1/dataset/ATIS/test.json"
    VAL_PROPORTION = 0.10

    train_raw, test_raw, dev_raw = get_data(TRAIN_DATA_PATH, TEST_DATA_PATH, VAL_PROPORTION)

    words = sum([x['utterance'].split() for x in train_raw], [])
    corpus = train_raw + dev_raw + test_raw
    slots = set(sum([line['slots'].split() for line in corpus],[]))
    intents = set([line['intent'] for line in corpus])

    # Create the vocabulary
    vocabulary = Lang(words, intents, slots, cutoff=0)

    # Create the dataset
    train_dataset = IntentsAndSlots(train_raw, vocabulary)
    dev_dataset = IntentsAndSlots(dev_raw, vocabulary)
    test_dataset = IntentsAndSlots(test_raw, vocabulary)

    # Data loaders
    train_loader = DataLoader(train_dataset, batch_size=128, collate_fn=collate_fn,  shuffle=True)
    dev_loader = DataLoader(dev_dataset, batch_size=64, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=64, collate_fn=collate_fn)

    # Define model parameters
    HID_SIZE = 200
    EMB_SIZE = 300
    OUT_SLOT = len(vocabulary.slot2id)
    OUT_INT = len(vocabulary.intent2id)
    VOCAB_LEN = len(vocabulary.word2id)

    # Define hyperparameters
    N_EPOCHS = 200
    RUNS = 5
    PATIENCE_INITIAL = 5
    CLIP = 5
    LR = 0.0001

    # Loss function
    criterion_slots = nn.CrossEntropyLoss(ignore_index=PAD_TOKEN)
    criterion_intents = nn.CrossEntropyLoss() # Because we do not have the pad token


    # Initialize the tokenizer
    model = ModelIAS(HID_SIZE,
                     OUT_SLOT,
                     OUT_INT,
                     EMB_SIZE,
                     VOCAB_LEN,
                     pad_index=PAD_TOKEN,
                     bidirectional=True,
                     dropout_flag=False).to(DEVICE)
    
    # Load the model's state dict onto the CPU
    model.load_state_dict(torch.load("NLU/part_1/bin/LSTM_bidir_run_1.pt", map_location=DEVICE))
    # Set the model to evaluation mode
    results_test, intent_test, _ = eval_loop(test_loader,
                                             criterion_slots,
                                             criterion_intents,
                                             model,
                                             vocabulary)
                                             
    print("Slot F1:", results_test['total']['f'])
    print("Intent Accuracy:", intent_test['accuracy'])

    # # ---------------------------------------------------------
    # # Ex 1.0: LSTM Model - Baseline
    # # ---------------------------------------------------------

    # # List to store the results for each run
    # slot_f1s, intent_acc = [], []

    # for run in tqdm(range(0, RUNS)):

    #     # Initialize the model
    #     model = ModelIAS(HID_SIZE,
    #                      OUT_SLOT,
    #                      OUT_INT,
    #                      EMB_SIZE,
    #                      VOCAB_LEN,
    #                      pad_index=PAD_TOKEN).to(DEVICE)
    #     model.apply(init_weights)

    #     # Optimizer
    #     optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)

    #     if LOG_WANDB:
    #         # Initialize a new WandB run for each learning rate
    #         MODEL_NAME = "LSTM_run_" + str(run)
    #         wandb.init(project=PROJECT_NAME, name=MODEL_NAME, reinit=True)
    #         # Log hyperparameters to WandB
    #         wandb.config.update({
    #             "model": MODEL_NAME,
    #         })

    #     # Training
    #     best_model = training(model,
    #                           vocabulary,
    #                           train_loader,
    #                           dev_loader,
    #                           criterion_slots,
    #                           criterion_intents,
    #                           optimizer,
    #                           N_EPOCHS,
    #                           PATIENCE_INITIAL,
    #                           CLIP,
    #                           LOG_WANDB)

    #     # Load the best model and evaluate on the test set
    #     best_model.to(DEVICE)
    #     results_test, intent_test, _ = eval_loop(test_loader,
    #                                              criterion_slots,
    #                                              criterion_intents,
    #                                              best_model,
    #                                              vocabulary)
    #     intent_acc.append(intent_test['accuracy'])
    #     slot_f1s.append(results_test['total']['f'])

    # # Print the average results
    # slot_f1s = np.asarray(slot_f1s)
    # intent_acc = np.asarray(intent_acc)
    # print('Slot F1', round(slot_f1s.mean(),3), '+-', round(slot_f1s.std(),3))
    # print('Intent Acc', round(intent_acc.mean(), 3), '+-', round(intent_acc.std(), 3))

    # avg_slot_f1 = round(slot_f1s.mean(),3)
    # std_slot_f1 = round(slot_f1s.std(), 3)
    # avg_intent_acc = round(intent_acc.mean(), 3)
    # std_intent_acc = round(intent_acc.std(), 3)

    # if LOG_WANDB:
    #     # Initialize a new WandB run for each learning rate
    #     MODEL_NAME = "AVG_LSTM"
    #     wandb.init(project=PROJECT_NAME, name=MODEL_NAME, reinit=True)
    #     # Log hyperparameters to WandB
    #     wandb.config.update({
    #         "model": MODEL_NAME,
    #     })
    #     wandb.log({"Slot F1 Evaluation": avg_slot_f1,
    #             "Slot F1 Std": std_slot_f1,
    #             "Intent Accuracy Evaluation": avg_intent_acc,
    #             "Intent Accuracy Std": std_intent_acc})
    #     # Finish the current WandB run
    #     wandb.finish()

    # # ---------------------------------------------------------
    # # Ex 1.1: Bidirectional LSTM Model
    # # ---------------------------------------------------------

    # # List to store the results for each run
    # slot_f1s, intent_acc = [], []

    # for run in tqdm(range(0, RUNS)):

    #     # Initialize the model
    #     model = ModelIAS(HID_SIZE,
    #                      OUT_SLOT,
    #                      OUT_INT,
    #                      EMB_SIZE,
    #                      VOCAB_LEN,
    #                      pad_index=PAD_TOKEN,
    #                      bidirectional=True).to(DEVICE)
    #     model.apply(init_weights)

    #     # Optimizer
    #     optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)

    #     if LOG_WANDB:
    #         # Initialize a new WandB run for each learning rate
    #         MODEL_NAME = "LSTM_bidir_run_" + str(run)
    #         wandb.init(project=PROJECT_NAME, name=MODEL_NAME, reinit=True)
    #         # Log hyperparameters to WandB
    #         wandb.config.update({
    #             "model": MODEL_NAME,
    #         })

    #     # Training
    #     best_model = training(model,
    #                           vocabulary,
    #                           train_loader,
    #                           dev_loader,
    #                           criterion_slots,
    #                           criterion_intents,
    #                           optimizer,
    #                           N_EPOCHS,
    #                           PATIENCE_INITIAL,
    #                           CLIP,
    #                           LOG_WANDB)

    #     # Load the best model and evaluate on the test set
    #     best_model.to(DEVICE)
    #     results_test, intent_test, _ = eval_loop(test_loader,
    #                                              criterion_slots,
    #                                              criterion_intents,
    #                                              best_model,
    #                                              vocabulary)
    #     intent_acc.append(intent_test['accuracy'])
    #     slot_f1s.append(results_test['total']['f'])


    # # Print the average results
    # slot_f1s = np.asarray(slot_f1s)
    # intent_acc = np.asarray(intent_acc)
    # print('Slot F1', round(slot_f1s.mean(),3), '+-', round(slot_f1s.std(),3))
    # print('Intent Acc', round(intent_acc.mean(), 3), '+-', round(intent_acc.std(), 3))

    # avg_slot_f1 = round(slot_f1s.mean(),3)
    # std_slot_f1 = round(slot_f1s.std(), 3)
    # avg_intent_acc = round(intent_acc.mean(), 3)
    # std_intent_acc = round(intent_acc.std(), 3)

    # if LOG_WANDB:
    #     # Initialize a new WandB run for each learning rate
    #     MODEL_NAME = "AVG_LSTM_bidir"
    #     wandb.init(project=PROJECT_NAME, name=MODEL_NAME, reinit=True)
    #     # Log hyperparameters to WandB
    #     wandb.config.update({
    #         "model": MODEL_NAME,
    #     })
    #     wandb.log({"Slot F1 Evaluation": avg_slot_f1,
    #             "Slot F1 Std": std_slot_f1,
    #             "Intent Accuracy Evaluation": avg_intent_acc,
    #             "Intent Accuracy Std": std_intent_acc})
    #     # Finish the current WandB run
    #     wandb.finish()

    # # ---------------------------------------------------------
    # # Ex 1.2: Bidirectional LSTM Model + Dropout
    # # ---------------------------------------------------------

    # # List to store the results for each run
    # slot_f1s, intent_acc = [], []

    # for run in tqdm(range(0, RUNS)):

    #     # Initialize the model
    #     model = ModelIAS(HID_SIZE,
    #                      OUT_SLOT,
    #                      OUT_INT,
    #                      EMB_SIZE,
    #                      VOCAB_LEN,
    #                      pad_index=PAD_TOKEN,
    #                      bidirectional=True,
    #                      dropout_flag=True).to(DEVICE)
    #     model.apply(init_weights)

    #     # Optimizer
    #     optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)

    #     if LOG_WANDB:
    #         # Initialize a new WandB run for each learning rate
    #         MODEL_NAME = "LSTM_bidir_Do_run_" + str(run)
    #         wandb.init(project=PROJECT_NAME, name=MODEL_NAME, reinit=True)
    #         # Log hyperparameters to WandB
    #         wandb.config.update({
    #             "model": MODEL_NAME,
    #         })

    #     # Training
    #     best_model = training(model,
    #                           vocabulary,
    #                           train_loader,
    #                           dev_loader,
    #                           criterion_slots,
    #                           criterion_intents,
    #                           optimizer,
    #                           N_EPOCHS,
    #                           PATIENCE_INITIAL,
    #                           CLIP,
    #                           LOG_WANDB)

    #     # Load the best model and evaluate on the test set
    #     best_model.to(DEVICE)
    #     results_test, intent_test, _ = eval_loop(test_loader,
    #                                              criterion_slots,
    #                                              criterion_intents,
    #                                              best_model,
    #                                              vocabulary)
    #     intent_acc.append(intent_test['accuracy'])
    #     slot_f1s.append(results_test['total']['f'])


    # # Print the average results
    # slot_f1s = np.asarray(slot_f1s)
    # intent_acc = np.asarray(intent_acc)
    # print('Slot F1', round(slot_f1s.mean(),3), '+-', round(slot_f1s.std(),3))
    # print('Intent Acc', round(intent_acc.mean(), 3), '+-', round(slot_f1s.std(), 3))

    # avg_slot_f1 = round(slot_f1s.mean(),3)
    # std_slot_f1 = round(slot_f1s.std(), 3)
    # avg_intent_acc = round(intent_acc.mean(), 3)
    # std_intent_acc = round(intent_acc.std(), 3)

    # if LOG_WANDB:
    #     # Initialize a new WandB run for each learning rate
    #     MODEL_NAME = "AVG_LSTM_bidir_Do"
    #     wandb.init(project=PROJECT_NAME, name=MODEL_NAME, reinit=True)
    #     # Log hyperparameters to WandB
    #     wandb.config.update({
    #         "model": MODEL_NAME,
    #     })
    #     wandb.log({"Slot F1 Evaluation": avg_slot_f1,
    #             "Slot F1 Std": std_slot_f1,
    #             "Intent Accuracy Evaluation": avg_intent_acc,
    #             "Intent Accuracy Std": std_intent_acc})
    #     # Finish the current WandB run
    #     wandb.finish()
