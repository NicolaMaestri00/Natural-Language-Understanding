''' This file is used to run the functions and print the results  '''


import os
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import BertTokenizer
import numpy as np
import wandb

from functions import training, eval_loop
from model import Bert_slot_intent_classifier
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

    # Load the dataset
    TRAIN_DATA_PATH = "NLU/part_2/dataset/ATIS/train.json"
    TEST_DATA_PATH = "NLU/part_2/dataset/ATIS/test.json"
    VAL_PROPORTION = 0.10

    train_raw, test_raw, dev_raw = get_data(TRAIN_DATA_PATH, TEST_DATA_PATH, VAL_PROPORTION)

    corpus = train_raw + dev_raw + test_raw
    slots = set(sum([line['slots'].split() for line in corpus],[]))
    intents = set([line['intent'] for line in corpus])

    # Create the vocabulary
    vocabulary = Lang(intents, slots)

    # Create our datasets
    train_dataset = IntentsAndSlots(train_raw, vocabulary)
    dev_dataset = IntentsAndSlots(dev_raw, vocabulary)
    test_dataset = IntentsAndSlots(test_raw, vocabulary)

    # Data loaders
    train_loader = DataLoader(train_dataset,
                              batch_size=128,
                              collate_fn=collate_fn,
                              shuffle=True)

    dev_loader = DataLoader(dev_dataset,
                            batch_size=64,
                            collate_fn=collate_fn)

    test_loader = DataLoader(test_dataset,
                             batch_size=64,
                             collate_fn=collate_fn)

    # Define hyperparameters
    N_RUNS = 1
    N_EPOCHS = 50
    PATIENCE = 5
    CLIP = 5

    # Loss function
    criterion_slots = nn.CrossEntropyLoss(ignore_index=PAD_TOKEN)
    criterion_intents = nn.CrossEntropyLoss()


    # Initialize the tokenizer
    PRETRAINED_MODEL_NAME = "bert-base-uncased"
    tokenizer = BertTokenizer.from_pretrained(PRETRAINED_MODEL_NAME)
    model = Bert_slot_intent_classifier(PRETRAINED_MODEL_NAME,
                                    num_intents=len(vocabulary.intent2id),
                                    num_slots=len(vocabulary.slot2id))
    
    # Load the model's state dict onto the CPU
    model.load_state_dict(torch.load("NLU/part_2/bin/bert_base_full_tuned_run_0.pt", map_location=DEVICE))
    # Set the model to evaluation mode
    model.eval()

    # Evaluate
    results_test, intent_test, _ = eval_loop(test_loader,
                                            criterion_slots,
                                            criterion_intents,
                                            model,
                                            vocabulary,
                                            tokenizer)

    print("Slot F1:", results_test['total']['f'])
    print("Intent Accuracy:", intent_test['accuracy'])


    # # ---------------------------------------------------------
    # #  Bert base - Fine-Tuning
    # # ---------------------------------------------------------

    # # Initialize the tokenizer
    # PRETRAINED_MODEL_NAME = "bert-base-uncased"
    # tokenizer = BertTokenizer.from_pretrained(PRETRAINED_MODEL_NAME)

    # # ---------------------------------------------------------
    # #  Full Fine-Tuning
    # # ---------------------------------------------------------

    # # List to store the results for each run
    # slot_f1s, intent_acc = [], []

    # for run in range(N_RUNS):

    #     # Initialize model
    #     model = Bert_slot_intent_classifier(PRETRAINED_MODEL_NAME,
    #                                         num_intents=len(vocabulary.intent2id),
    #                                         num_slots=len(vocabulary.slot2id))
    #     model.to(DEVICE)

    #     # Optimizer
    #     optimizer = optim.AdamW([
    #         {'params': model.bert.parameters(), 'lr': 5e-5, 'weight_decay': 0.01},              # lr for BERT parameters
    #         {'params': model.intent_classifier.parameters(), 'lr': 1e-4, 'weight_decay': 0.01}, # lr for intent classifier
    #         {'params': model.slot_classifier.parameters(), 'lr': 1e-4, 'weight_decay': 0.01}    # lr for slot classifier
    #     ])

    #     if LOG_WANDB:
    #         # Initialize a new WandB run
    #         MODEL_NAME = "Bert_base_full_tuned_run_" + str(run)
    #         wandb.init(project=PROJECT_NAME, name=MODEL_NAME, reinit=True)
    #         # Log hyperparameters to WandB
    #         wandb.config.update({
    #             "model": MODEL_NAME,
    #         })

    #     # Training
    #     training(model,
    #             vocabulary,
    #             tokenizer,
    #             train_loader,
    #             dev_loader,
    #             optimizer,
    #             criterion_slots,
    #             criterion_intents,
    #             n_epochs=N_EPOCHS,
    #             init_patience=5,
    #             clip = 5,
    #             log_wandb=LOG_WANDB)

    #     # Evaluate
    #     results_test, intent_test, _ = eval_loop(test_loader,
    #                                             criterion_slots,
    #                                             criterion_intents,
    #                                             model,
    #                                             vocabulary,
    #                                             tokenizer)

    #     intent_acc.append(intent_test['accuracy'])
    #     slot_f1s.append(results_test['total']['f'])


    # # Print the average results
    # slot_f1s = np.asarray(slot_f1s)
    # intent_acc = np.asarray(intent_acc)
    # print('Slot F1', round(slot_f1s.mean(),3), '+-', round(slot_f1s.std(),3))
    # print('Intent Acc', round(intent_acc.mean(), 3), '+-', round(intent_acc.std(), 3))

    # avg_slot_f1 = round(slot_f1s.mean(),3)
    # std_slot_f1 = round(slot_f1s.std(),3)
    # avg_intent_acc = round(intent_acc.mean(), 3)
    # std_intent_acc = round(intent_acc.std(), 3)

    # if LOG_WANDB:
    #     # Initialize a new WandB run for the average results
    #     MODEL_NAME = "AVG_Bert_base_full_tuned"
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
    # #  Half model Fine-Tuning
    # # ---------------------------------------------------------

    # # List to store the results for each run
    # slot_f1s, intent_acc = [], []

    # for run in range(N_RUNS):

    #     # Initialize model
    #     model = Bert_slot_intent_classifier(PRETRAINED_MODEL_NAME,
    #                                         num_intents=len(vocabulary.intent2id),
    #                                         num_slots=len(vocabulary.slot2id))
    #     model.to(DEVICE)

    #     # Freeze BERT layers but not the classifier layers
    #     for name, param in model.named_parameters():
    #         param.requires_grad = True
    #         if not name.startswith(("bert.encoder.layer.6",
    #                                 "bert.encoder.layer.7",
    #                                 "bert.encoder.layer.8",
    #                                 "bert.encoder.layer.9",
    #                                 "bert.encoder.layer.10",
    #                                 "bert.encoder.layer.11",
    #                                 "bert.pooler",
    #                                 "intent_classifier",
    #                                 "slot_classifier")):
    #             param.requires_grad = False

    #     # Optimizer
    #     optimizer = optim.AdamW([
    #         {'params': model.bert.parameters(), 'lr': 5e-5, 'weight_decay': 0.01},              # lr for BERT parameters
    #         {'params': model.intent_classifier.parameters(), 'lr': 1e-4, 'weight_decay': 0.01}, # lr for intent classifier
    #         {'params': model.slot_classifier.parameters(), 'lr': 1e-4, 'weight_decay': 0.01}    # lr for slot classifier
    #     ])

    #     if LOG_WANDB:
    #         # Initialize a new WandB run
    #         MODEL_NAME = "Bert_base_half_tuned_run_" + str(run)
    #         wandb.init(project=PROJECT_NAME, name=MODEL_NAME, reinit=True)
    #         # Log hyperparameters to WandB
    #         wandb.config.update({
    #             "model": MODEL_NAME,
    #         })

    #     # Training
    #     training(model,
    #             vocabulary,
    #             tokenizer,
    #             train_loader,
    #             dev_loader,
    #             optimizer,
    #             criterion_slots,
    #             criterion_intents,
    #             n_epochs=N_EPOCHS,
    #             init_patience=5,
    #             clip = 5,
    #             log_wandb=LOG_WANDB)

    #     # Evaluate
    #     results_test, intent_test, _ = eval_loop(test_loader,
    #                                             criterion_slots,
    #                                             criterion_intents,
    #                                             model,
    #                                             vocabulary,
    #                                             tokenizer)

    #     intent_acc.append(intent_test['accuracy'])
    #     slot_f1s.append(results_test['total']['f'])

    # # Print the average results
    # slot_f1s = np.asarray(slot_f1s)
    # intent_acc = np.asarray(intent_acc)
    # print('Slot F1', round(slot_f1s.mean(),3), '+-', round(slot_f1s.std(),3))
    # print('Intent Acc', round(intent_acc.mean(), 3), '+-', round(intent_acc.std(), 3))

    # avg_slot_f1 = round(slot_f1s.mean(),3)
    # std_slot_f1 = round(slot_f1s.std(),3)
    # avg_intent_acc = round(intent_acc.mean(), 3)
    # std_intent_acc = round(intent_acc.std(), 3)

    # if LOG_WANDB:
    #     # Initialize a new WandB run for the average results
    #     MODEL_NAME = "AVG_Bert_base_half_tuned"
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
    # #  Last Layer Fine-Tuning
    # # ---------------------------------------------------------

    # # List to store the results for each run
    # slot_f1s, intent_acc = [], []

    # for run in range(N_RUNS):

    #     # Initialize model
    #     model = Bert_slot_intent_classifier(PRETRAINED_MODEL_NAME,
    #                                         num_intents=len(vocabulary.intent2id),
    #                                         num_slots=len(vocabulary.slot2id))
    #     model.to(DEVICE)

    #     # Freeze BERT layers but not the classifier layers
    #     for name, param in model.named_parameters():
    #         param.requires_grad = True
    #         if not name.startswith(("bert.encoder.layer.11",
    #                                 "bert.pooler",
    #                                 "intent_classifier",
    #                                 "slot_classifier")):
    #             param.requires_grad = False

    #     # Optimizer
    #     optimizer = optim.AdamW([
    #         {'params': model.bert.parameters(), 'lr': 5e-5, 'weight_decay': 0.01},              # lr for BERT parameters
    #         {'params': model.intent_classifier.parameters(), 'lr': 1e-4, 'weight_decay': 0.01}, # lr for intent classifier
    #         {'params': model.slot_classifier.parameters(), 'lr': 1e-4, 'weight_decay': 0.01}    # lr for slot classifier
    #     ])

    #     if LOG_WANDB:
    #         # Initialize a new WandB run
    #         MODEL_NAME = "Bert_base_last_layer_tuned_run_" + str(run)
    #         wandb.init(project=PROJECT_NAME, name=MODEL_NAME, reinit=True)
    #         # Log hyperparameters to WandB
    #         wandb.config.update({
    #             "model": MODEL_NAME,
    #         })

    #     # Training
    #     training(model,
    #             vocabulary,
    #             tokenizer,
    #             train_loader,
    #             dev_loader,
    #             optimizer,
    #             criterion_slots,
    #             criterion_intents,
    #             n_epochs=N_EPOCHS,
    #             init_patience=5,
    #             clip = 5,
    #             log_wandb=LOG_WANDB)

    #     # Evaluate
    #     results_test, intent_test, _ = eval_loop(test_loader,
    #                                             criterion_slots,
    #                                             criterion_intents,
    #                                             model,
    #                                             vocabulary,
    #                                             tokenizer)

    #     intent_acc.append(intent_test['accuracy'])
    #     slot_f1s.append(results_test['total']['f'])

    # # Print the average results
    # slot_f1s = np.asarray(slot_f1s)
    # intent_acc = np.asarray(intent_acc)
    # print('Slot F1', round(slot_f1s.mean(),3), '+-', round(slot_f1s.std(),3))
    # print('Intent Acc', round(intent_acc.mean(), 3), '+-', round(intent_acc.std(), 3))

    # avg_slot_f1 = round(slot_f1s.mean(),3)
    # std_slot_f1 = round(slot_f1s.std(),3)
    # avg_intent_acc = round(intent_acc.mean(), 3)
    # std_intent_acc = round(intent_acc.std(), 3)

    # if LOG_WANDB:
    #     # Initialize a new WandB run for the average results
    #     MODEL_NAME = "AVG_Bert_base_last_layer_tuned"
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
    # #  Bert Large - Fine-Tuning
    # # ---------------------------------------------------------

    # # Initialize the tokenizer
    # PRETRAINED_MODEL_NAME = "bert-large-uncased"
    # tokenizer = BertTokenizer.from_pretrained(PRETRAINED_MODEL_NAME)

    # # ---------------------------------------------------------
    # #  Full Fine-Tuning
    # # ---------------------------------------------------------

    # # List to store the results for each run
    # slot_f1s, intent_acc = [], []

    # for run in range(N_RUNS):

    #     # Initialize model
    #     model = Bert_slot_intent_classifier(PRETRAINED_MODEL_NAME,
    #                                         num_intents=len(vocabulary.intent2id),
    #                                         num_slots=len(vocabulary.slot2id))

    #     model.to(DEVICE)

    #     # Optimizer
    #     optimizer = optim.AdamW([
    #         {'params': model.bert.parameters(), 'lr': 5e-5, 'weight_decay': 0.01},              # lr for BERT parameters
    #         {'params': model.intent_classifier.parameters(), 'lr': 1e-4, 'weight_decay': 0.01}, # lr for intent classifier
    #         {'params': model.slot_classifier.parameters(), 'lr': 1e-4, 'weight_decay': 0.01}    # lr for slot classifier
    #     ])

    #     if LOG_WANDB:
    #         # Initialize a new WandB run
    #         MODEL_NAME = "Bert_large_full_tuned_run_" + str(run)
    #         wandb.init(project=PROJECT_NAME, name=MODEL_NAME, reinit=True)
    #         # Log hyperparameters to WandB
    #         wandb.config.update({
    #             "model": MODEL_NAME,
    #         })

    #     # Training
    #     training(model,
    #             vocabulary,
    #             tokenizer,
    #             train_loader,
    #             dev_loader,
    #             optimizer,
    #             criterion_slots,
    #             criterion_intents,
    #             n_epochs=N_EPOCHS,
    #             init_patience=5,
    #             clip = 5,
    #             log_wandb=LOG_WANDB)

    #     # Evaluate
    #     results_test, intent_test, _ = eval_loop(test_loader,
    #                                             criterion_slots,
    #                                             criterion_intents,
    #                                             model,
    #                                             vocabulary,
    #                                             tokenizer)

    #     intent_acc.append(intent_test['accuracy'])
    #     slot_f1s.append(results_test['total']['f'])


    # # Print the average results
    # slot_f1s = np.asarray(slot_f1s)
    # intent_acc = np.asarray(intent_acc)
    # print('Slot F1', round(slot_f1s.mean(),3), '+-', round(slot_f1s.std(),3))
    # print('Intent Acc', round(intent_acc.mean(), 3), '+-', round(intent_acc.std(), 3))

    # avg_slot_f1 = round(slot_f1s.mean(),3)
    # std_slot_f1 = round(slot_f1s.std(),3)
    # avg_intent_acc = round(intent_acc.mean(), 3)
    # std_intent_acc = round(intent_acc.std(), 3)

    # if LOG_WANDB:
    #     # Initialize a new WandB run for the average results
    #     MODEL_NAME = "AVG_Bert_large_full_tuned"
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
    # #  Half model Fine-Tuning
    # # ---------------------------------------------------------

    # # List to store the results for each run
    # slot_f1s, intent_acc = [], []

    # for run in range(N_RUNS):

    #     # Initialize model
    #     model = Bert_slot_intent_classifier(PRETRAINED_MODEL_NAME,
    #                                         num_intents=len(vocabulary.intent2id),
    #                                         num_slots=len(vocabulary.slot2id))
    #     model.to(DEVICE)

    #     # Freeze BERT layers but not the classifier layers
    #     for name, param in model.named_parameters():
    #         param.requires_grad = True
    #         if not name.startswith(("bert.encoder.layer.12",
    #                                 "bert.encoder.layer.13",
    #                                 "bert.encoder.layer.14",
    #                                 "bert.encoder.layer.15",
    #                                 "bert.encoder.layer.16",
    #                                 "bert.encoder.layer.17",
    #                                 "bert.encoder.layer.18",
    #                                 "bert.encoder.layer.19",
    #                                 "bert.encoder.layer.20",
    #                                 "bert.encoder.layer.21",
    #                                 "bert.encoder.layer.22",
    #                                 "bert.encoder.layer.23",
    #                                 "bert.pooler",
    #                                 "intent_classifier",
    #                                 "slot_classifier")):
    #             param.requires_grad = False

    #     # Optimizer
    #     optimizer = optim.AdamW([
    #         {'params': model.bert.parameters(), 'lr': 5e-5, 'weight_decay': 0.01},              # lr for BERT parameters
    #         {'params': model.intent_classifier.parameters(), 'lr': 1e-4, 'weight_decay': 0.01}, # lr for intent classifier
    #         {'params': model.slot_classifier.parameters(), 'lr': 1e-4, 'weight_decay': 0.01}    # lr for slot classifier
    #     ])

    #     if LOG_WANDB:
    #         # Initialize a new WandB run
    #         MODEL_NAME = "Bert_large_half_tuned_run_" + str(run)
    #         wandb.init(project=PROJECT_NAME, name=MODEL_NAME, reinit=True)
    #         # Log hyperparameters to WandB
    #         wandb.config.update({
    #             "model": MODEL_NAME,
    #         })

    #     # Training
    #     training(model,
    #             vocabulary,
    #             tokenizer,
    #             train_loader,
    #             dev_loader,
    #             optimizer,
    #             criterion_slots,
    #             criterion_intents,
    #             n_epochs=N_EPOCHS,
    #             init_patience=5,
    #             clip = 5,
    #             log_wandb=LOG_WANDB)

    #     # Evaluate
    #     results_test, intent_test, _ = eval_loop(test_loader,
    #                                             criterion_slots,
    #                                             criterion_intents,
    #                                             model,
    #                                             vocabulary,
    #                                             tokenizer)

    #     intent_acc.append(intent_test['accuracy'])
    #     slot_f1s.append(results_test['total']['f'])

    # # Print the average results
    # slot_f1s = np.asarray(slot_f1s)
    # intent_acc = np.asarray(intent_acc)
    # print('Slot F1', round(slot_f1s.mean(),3), '+-', round(slot_f1s.std(),3))
    # print('Intent Acc', round(intent_acc.mean(), 3), '+-', round(intent_acc.std(), 3))

    # avg_slot_f1 = round(slot_f1s.mean(),3)
    # std_slot_f1 = round(slot_f1s.std(),3)
    # avg_intent_acc = round(intent_acc.mean(), 3)
    # std_intent_acc = round(intent_acc.std(), 3)

    # if LOG_WANDB:
    #     # Initialize a new WandB run for the average results
    #     MODEL_NAME = "AVG_Bert_large_half_tuned"
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
    # #  Last Layer Fine-Tuning
    # # ---------------------------------------------------------

    # # List to store the results for each run
    # slot_f1s, intent_acc = [], []

    # for run in range(N_RUNS):

    #     # Initialize model
    #     model = Bert_slot_intent_classifier(PRETRAINED_MODEL_NAME,
    #                                         num_intents=len(vocabulary.intent2id),
    #                                         num_slots=len(vocabulary.slot2id))
    #     model.to(DEVICE)

    #     # Freeze BERT layers but not the classifier layers
    #     for name, param in model.named_parameters():
    #         param.requires_grad = True
    #         if not name.startswith(("bert.encoder.layer.23",
    #                                 "bert.pooler",
    #                                 "intent_classifier",
    #                                 "slot_classifier")):
    #             param.requires_grad = False

    #     # Optimizer
    #     optimizer = optim.AdamW([
    #         {'params': model.bert.parameters(), 'lr': 5e-5, 'weight_decay': 0.01},              # lr for BERT parameters
    #         {'params': model.intent_classifier.parameters(), 'lr': 1e-4, 'weight_decay': 0.01}, # lr for intent classifier
    #         {'params': model.slot_classifier.parameters(), 'lr': 1e-4, 'weight_decay': 0.01}    # lr for slot classifier
    #     ])

    #     if LOG_WANDB:
    #         # Initialize a new WandB run
    #         MODEL_NAME = "Bert_large_last_layer_tuned_run_" + str(run)
    #         wandb.init(project=PROJECT_NAME, name=MODEL_NAME, reinit=True)
    #         # Log hyperparameters to WandB
    #         wandb.config.update({
    #             "model": MODEL_NAME,
    #         })

    #     # Training
    #     training(model,
    #             vocabulary,
    #             tokenizer,
    #             train_loader,
    #             dev_loader,
    #             optimizer,
    #             criterion_slots,
    #             criterion_intents,
    #             n_epochs=N_EPOCHS,
    #             init_patience=5,
    #             clip = 5,
    #             log_wandb=LOG_WANDB)

    #     # Evaluate
    #     results_test, intent_test, _ = eval_loop(test_loader,
    #                                             criterion_slots,
    #                                             criterion_intents,
    #                                             model,
    #                                             vocabulary,
    #                                             tokenizer)

    #     intent_acc.append(intent_test['accuracy'])
    #     slot_f1s.append(results_test['total']['f'])

    # # Print the average results
    # slot_f1s = np.asarray(slot_f1s)
    # intent_acc = np.asarray(intent_acc)
    # print('Slot F1', round(slot_f1s.mean(),3), '+-', round(slot_f1s.std(),3))
    # print('Intent Acc', round(intent_acc.mean(), 3), '+-', round(intent_acc.std(), 3))

    # avg_slot_f1 = round(slot_f1s.mean(),3)
    # std_slot_f1 = round(slot_f1s.std(),3)
    # avg_intent_acc = round(intent_acc.mean(), 3)
    # std_intent_acc = round(intent_acc.std(), 3)

    # if LOG_WANDB:
    #     # Initialize a new WandB run for the average results
    #     MODEL_NAME = "AVG_Bert_large_last_layer_tuned"
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
