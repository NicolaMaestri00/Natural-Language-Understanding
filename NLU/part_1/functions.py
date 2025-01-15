''' This file contains the functions that are used in the training and evaluation loops '''


from conll import evaluate
from sklearn.metrics import classification_report
import torch
import numpy as np
import copy
import wandb


def train_loop(data, optimizer, criterion_slots, criterion_intents, model, clip=5):
    ''' Training loop for the model '''
    model.train()
    loss_array = []
    for sample in data:
        optimizer.zero_grad() # Zeroing the gradient
        slots, intent = model(sample['utterances'], sample['slots_len'])
        loss_intent = criterion_intents(intent, sample['intents'])
        loss_slot = criterion_slots(slots, sample['y_slots'])
        loss = loss_intent + loss_slot # In joint training we sum the losses.
                                       # Is there another way to do that?
        loss_array.append(loss.item())
        loss.backward() # Compute the gradient, deleting the computational graph
        # clip the gradient to avoid exploding gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step() # Update the weights
    return loss_array


def eval_loop(data, criterion_slots, criterion_intents, model, lang):
    ''' Evaluation loop for the model '''
    model.eval()
    loss_array = []

    ref_intents = []
    hyp_intents = []

    ref_slots = []
    hyp_slots = []
    #softmax = nn.Softmax(dim=1) # Use Softmax if you need the actual probability
    with torch.no_grad(): # It used to avoid the creation of computational graph
        for sample in data:
            slots, intents = model(sample['utterances'], sample['slots_len'])
            loss_intent = criterion_intents(intents, sample['intents'])
            loss_slot = criterion_slots(slots, sample['y_slots'])
            loss = loss_intent + loss_slot
            loss_array.append(loss.item())
            # Intent inference
            # Get the highest probable class
            out_intents = [lang.id2intent[x]
                           for x in torch.argmax(intents, dim=1).tolist()]
            gt_intents = [lang.id2intent[x] for x in sample['intents'].tolist()]
            ref_intents.extend(gt_intents)
            hyp_intents.extend(out_intents)

            # Slot inference
            output_slots = torch.argmax(slots, dim=1)
            for id_seq, seq in enumerate(output_slots):
                length = sample['slots_len'].tolist()[id_seq]
                utt_ids = sample['utterance'][id_seq][:length].tolist()
                gt_ids = sample['y_slots'][id_seq].tolist()
                gt_slots = [lang.id2slot[elem] for elem in gt_ids[:length]]
                utterance = [lang.id2word[elem] for elem in utt_ids]
                to_decode = seq[:length].tolist()
                ref_slots.append([(utterance[id_el], elem) for id_el, elem in enumerate(gt_slots)])
                tmp_seq = []
                for id_el, elem in enumerate(to_decode):
                    tmp_seq.append((utterance[id_el], lang.id2slot[elem]))
                hyp_slots.append(tmp_seq)
    try:
        results = evaluate(ref_slots, hyp_slots)
    except Exception as ex:
        # Sometimes the model predicts a class that is not in REF
        print("Warning:", ex)
        ref_s = set([x[1] for x in ref_slots])
        hyp_s = set([x[1] for x in hyp_slots])
        print(hyp_s.difference(ref_s))
        results = {"total":{"f":0}}

    report_intent = classification_report(ref_intents, hyp_intents,
                                          zero_division=False, output_dict=True)
    return results, report_intent, loss_array


def training(
        model,
        lang,
        train_loader,
        dev_loader,
        criterion_slots,
        criterion_intents,
        optimizer,
        n_epochs,
        init_patience,
        clip,
        log_wandb=False
        ):
    ''' This function is used to train the model '''

    patience = init_patience
    losses_train = []
    losses_dev = []
    sampled_epochs = []
    best_score = 0
    best_model = None

    for e in range(1,n_epochs):
        loss = train_loop(train_loader, optimizer, criterion_slots,
                          criterion_intents, model, clip)
        if e % 5 == 0:
            sampled_epochs.append(e)
            losses_train.append(np.asarray(loss).mean())
            results_dev, intent_res, loss_dev = eval_loop(dev_loader, criterion_slots,
                                                          criterion_intents, model, lang)
            losses_dev.append(np.asarray(loss_dev).mean())
            f1 = results_dev['total']['f']
            intent_accuracy = intent_res['accuracy']
            avg_score = (f1 + intent_accuracy)/2

            if log_wandb:
                # Log training loss and perplexity to WandB
                wandb.log({ "Training_Loss": losses_train[-1],
                            "Validation_Loss": losses_dev[-1],
                            "Slot F1": results_dev['total']['f'],
                            "Intent Accuracy": intent_res['accuracy']},
                            step=e)

            if avg_score > best_score:
                best_score = avg_score
                best_model = copy.deepcopy(model).to('cpu')
            else:
                patience -= 1
            if patience <= 0:       # Early stopping with patient
                break

    if log_wandb:
        # Finish the current WandB run
        wandb.finish()

    return best_model
