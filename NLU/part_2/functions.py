''' This file contains the functions that are used in the training and evaluation loops '''

from sklearn.metrics import classification_report
from conll import evaluate
from tqdm import tqdm
import numpy as np
import torch
import wandb

def train_loop(data, optimizer, criterion_slots, criterion_intents, model, clip=5):
    ''' This function is used to train the model '''
    model.train()
    loss_array = []
    for sample in data:
        optimizer.zero_grad() # Zeroing the gradient
        intent_logits, slot_logits  = model(sample['utterances'])
        loss_intent = criterion_intents(intent_logits, sample['intents'])
        loss_slot = criterion_slots(slot_logits, sample['y_slots'])
        loss = loss_intent + loss_slot
        loss_array.append(loss.item())
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()
    return loss_array


def eval_loop(data, criterion_slots, criterion_intents, model, lang, tokenizer):
    ''' This function is used to evaluate the model '''
    model.eval()
    loss_array = []

    ref_intents = []
    hyp_intents = []

    ref_slots = []
    hyp_slots = []

    #softmax = nn.Softmax(dim=1) # Use Softmax if you need the actual probability
    with torch.no_grad(): # It used to avoid the creation of computational graph
        for sample in data:
            intent_logits, slot_logits  = model(sample['utterances'])
            loss_intent = criterion_intents(intent_logits, sample['intents'])
            loss_slot = criterion_slots(slot_logits, sample['y_slots'])
            loss = loss_intent + loss_slot
            loss_array.append(loss.item())
            # Intent inference
            # Get the highest probable class
            out_intents = [lang.id2intent[x]
                           for x in torch.argmax(intent_logits, dim=1).tolist()]
            gt_intents = [lang.id2intent[x] for x in sample['intents'].tolist()]
            ref_intents.extend(gt_intents)
            hyp_intents.extend(out_intents)

            # Slot inference
            output_slots = torch.argmax(slot_logits, dim=1)
            for id_seq, seq in enumerate(output_slots):
                length = sample['slots_len'].tolist()[id_seq]
                utt_ids = sample['utterance'][id_seq][:length].tolist()
                gt_ids = sample['y_slots'][id_seq].tolist()
                gt_slots = [lang.id2slot[elem] for elem in gt_ids[:length]]
                utterance = [tokenizer.convert_ids_to_tokens(x) for x in utt_ids]
                to_decode = seq[:length].tolist()
                ref_slots.append([(utterance[id_el], elem) for id_el, elem in enumerate(gt_slots)])
                tmp_seq = []
                for id_el, elem in enumerate(to_decode):
                    tmp_seq.append((utterance[id_el], lang.id2slot[elem]))
                hyp_slots.append(tmp_seq)

    new_ref = []
    new_hyp = []
    for i, x in enumerate(ref_slots):
        tmp_ref = []
        tmp_hyp = []
        for j, y in enumerate(x):
            if y[1] != 'pad':
                tmp_ref.append(y)
                tmp_hyp.append(hyp_slots[i][j])
        new_ref.append(tmp_ref)
        new_hyp.append(tmp_hyp)

    try:
        results = evaluate(new_ref, new_hyp)
    except Exception as ex:
        # Sometimes the model predicts a class that is not in REF
        print("Warning:", ex)
        ref_s = []
        for x in ref_slots:
            for y in x:
                ref_s.append(y[1])
        ref_s = set(ref_s)
        hyp_s = []
        for x in hyp_slots:
            for y in x:
                hyp_s.append(y[1])
        hyp_s = set(hyp_s)
        print(len(ref_s), ref_s)
        print(len(hyp_s), hyp_s)
        print(hyp_s.difference(ref_s))
        results = {"total":{"f":0}}

    report_intent = classification_report(ref_intents, hyp_intents,
                                          zero_division=False, output_dict=True)
    return results, report_intent, loss_array


def training(model,
             lang,
             tokenizer,
             train_loader,
             dev_loader,
             optimizer,
             criterion_slots,
             criterion_intents,
             n_epochs=100,
             init_patience=3,
             clip = 5,
             log_wandb=False):
    ''' This function is used to train the model '''

    losses_train = []
    losses_dev = []
    sampled_epochs = []
    best_score = 0
    patience = init_patience
    finetuned_parameters = {}

    for e in tqdm(range(1,n_epochs+1)):
        loss = train_loop(train_loader, optimizer, criterion_slots, criterion_intents, model, clip)

        if e % 5 == 0:                                     # We check the performance every 5 epochs
            sampled_epochs.append(e)
            losses_train.append(np.asarray(loss).mean())
            results_dev, intent_res, loss_dev = eval_loop(dev_loader, criterion_slots,
                                                        criterion_intents, model, lang, tokenizer)
            losses_dev.append(np.asarray(loss_dev).mean())

            if log_wandb:
                # Log training loss and perplexity to WandB
                wandb.log({ "Training_Loss": losses_train[-1],
                            "Validation_Loss": losses_dev[-1],
                            "Slot F1": results_dev['total']['f'],
                            "Intent Accuracy": intent_res['accuracy']},
                            step=e)

            f1 = results_dev['total']['f']
            intent_accuracy = intent_res['accuracy']
            avg_score = (f1 + intent_accuracy)/2

            if avg_score > best_score:
                best_score = avg_score
                patience = init_patience
                # Save the best finetuned parameters
                for name, param in model.named_parameters():
                    if param.requires_grad:
                        finetuned_parameters[name] = param.data.cpu().numpy()
            else:
                patience -= 1
            if patience <= 0:       # Early stopping
                break

    if log_wandb:
        # Finish the run
        wandb.finish()

    # Load the best finetuned parameters
    for name, param in model.named_parameters():
        if name in finetuned_parameters:
            param.data = torch.tensor(finetuned_parameters[name]).to(param.device)

    device = 'cuda' if next(model.parameters()).is_cuda else 'cpu'
    model.to(device)
    model.eval()
