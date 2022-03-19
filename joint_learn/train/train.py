import torch
from torch import nn
from sklearn.metrics import accuracy_score
import logging
from tqdm import tqdm
from torch.autograd import Variable
import transformers
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data.sampler import RandomSampler
import numpy as np

logging.basicConfig(level=logging.INFO)

"""
Script for training the neural network and saving the better models 
while monitoring a metric like accuracy etc
"""


def train_model(model, optimizer, dataloaders, max_epochs, config_dict):
    device = config_dict["device"]
    criterion = nn.CrossEntropyLoss()  ## since we are doing binary classification
    max_accuracy = 5e-1
    for epoch in tqdm(range(max_epochs)):
        logging.info("Epoch: {}".format(epoch))
        y_true = list()
        y_pred = list()
        total_loss = 0
        for dataloader, data in dataloaders:
            for (
                batch,
                targets,
                lengths,
                raw_text,
                dataset_name,
            ) in dataloader["train_loader"]:
                batch, targets, lengths = data.sort_batch(batch, targets, lengths)

                model.zero_grad()
                pred = None
                loss = None

                ## perform forward pass
                if config_dict["model_name"] == "jl_lstm_attention":
                    pred, annotation_weight_matrix = model(
                        batch.to(device),
                        lengths.to(device),
                        dataset_name,
                    )
                    attention_loss = attention_penalty_loss(
                        annotation_weight_matrix,
                        config_dict["self_attention_config"]["penalty"],
                        device,
                    )

                    ## compute loss
                    loss = (
                        criterion(
                            pred.to(device), torch.autograd.Variable(targets).to(device)
                        )
                        + attention_loss
                    )
                else:
                    pred = model(
                        batch.to(device),
                        lengths.to(device),
                        dataset_name,
                    )
                    ## compute loss
                    loss = criterion(
                        pred.to(device),
                        torch.autograd.Variable(targets).to(device),
                    )

                ## perform backward pass
                loss.backward()

                ## update weights
                optimizer.step()

                ## accumulate targets from batch
                y_true += list(targets.numpy())

                ## accumulate preds from batch
                y_pred += list(
                    np.argmax(pred.data.float().detach().cpu().numpy(), axis=1)
                )

                ## accumulate train loss
                total_loss += loss

                # print(y_true, y_pred)

        ## computing accuracy using sklearn's function
        acc = accuracy_score(y_true, y_pred)

        ## compute model metrics on dev set
        val_acc, val_loss = evaluate_dev_set(
            model, criterion, dataloaders, config_dict, device
        )

        if val_acc > max_accuracy:
            max_accuracy = val_acc
            logging.info(
                "new model saved"
            )  ## save the model if it is better than the prior best
            torch.save(model.state_dict(), "{}.pth".format(config_dict["model_name"]))

        logging.info(
            "Train loss: {} - acc: {} -- Validation loss: {} - acc: {}".format(
                torch.mean(total_loss.data.float()), acc, val_loss, val_acc
            )
        )
    return model


def evaluate_dev_set(model, criterion, data_loaders, config_dict, device):
    """
    Evaluates the model performance on dev data
    """
    logging.info("Evaluating accuracy on dev set")

    y_true = list()
    y_pred = list()
    total_loss = 0
    for dataloader, data in data_loaders:
        for (
            batch,
            targets,
            lengths,
            raw_text,
            dataset_name,
        ) in dataloader["val_loader"]:
            batch, targets, lengths = data.sort_batch(batch, targets, lengths)
            ## perform forward pass
            pred = None
            loss = None

            ## perform forward pass
            if config_dict["model_name"] == "jl_lstm_attention":
                (pred, annotation_weight_matrix,) = model(
                    batch.to(device),
                    lengths.to(device),
                    dataset_name,
                )
                attention_loss = attention_penalty_loss(
                    annotation_weight_matrix,
                    config_dict["self_attention_config"]["penalty"],
                    device,
                )

                ## compute loss
                loss = (
                    criterion(
                        pred.to(device), torch.autograd.Variable(targets).to(device)
                    )
                    + attention_loss
                )
            else:
                pred = model(
                    batch.to(device),
                    lengths.to(device),
                    dataset_name,
                )

                ## compute loss
                loss = criterion(
                    pred.to(device), torch.autograd.Variable(targets).to(device)
                )

            y_true += list(targets)
            y_pred += list(np.argmax(pred.data.float().detach().cpu().numpy(), axis=1))
            total_loss += loss
    ## computing accuracy using sklearn's function
    acc = accuracy_score(y_true, y_pred)

    return acc, torch.mean(total_loss.data.float())


def attention_penalty_loss(annotation_weight_matrix, penalty_coef, device):
    """
    This function computes the loss from annotation/attention matrix
    to reduce redundancy in annotation matrix and for attention
    to focus on different parts of the sequence corresponding to the
    penalty term 'P' in the ICLR paper
    ----------------------------------
    'annotation_weight_matrix' refers to matrix 'A' in the ICLR paper
    annotation_weight_matrix shape: (batch_size, attention_out, seq_len)
    """
    batch_size, attention_out_size = annotation_weight_matrix.size(
        0
    ), annotation_weight_matrix.size(1)
    ## this fn computes ||AAT - I|| where norm is the frobenius norm
    ## taking transpose of annotation matrix
    ## shape post transpose: (batch_size, seq_len, attention_out)
    annotation_weight_matrix_trans = annotation_weight_matrix.transpose(1, 2)

    ## corresponds to AAT
    ## shape: (batch_size, attention_out, attention_out)
    annotation_mul = torch.bmm(annotation_weight_matrix, annotation_weight_matrix_trans)

    ## corresponds to 'I'
    identity = torch.eye(annotation_weight_matrix.size(1))
    ## make equal to the shape of annotation_mul and move it to device
    identity = Variable(
        identity.unsqueeze(0)
        .expand(batch_size, attention_out_size, attention_out_size)
        .to(device)
    )

    ## compute AAT - I
    annotation_mul_difference = annotation_mul - identity

    ## compute the frobenius norm
    penalty = frobenius_norm(annotation_mul_difference)

    ## compute loss
    loss = (penalty_coef * penalty / batch_size).type(torch.FloatTensor)

    return loss


def frobenius_norm(annotation_mul_difference):
    """
    Computes the frobenius norm of the annotation_mul_difference input as matrix
    """
    return torch.sum(
        torch.sum(torch.sum(annotation_mul_difference ** 2, 1), 1) ** 0.5
    ).type(torch.DoubleTensor)
