import torch
from torch.nn import functional
import os
import torch.nn as nn

def masked_cross_entropy(logits, target, mask):
    """
    Args:
        logits: A Variable containing a FloatTensor of size
            (batch, max_len, num_classes) which contains the
            unnormalized probability for each class.
        target: A Variable containing a LongTensor of size
            (batch, max_len) which contains the index of the true
            class for each corresponding step.
        length: A Variable containing a LongTensor of size (batch,)
            which contains the length of each data in a batch.

    Returns:
        loss: An average loss value masked by the length.
    """
    #mask = mask.transpose(0, 1).float()
    length = torch.sum(mask, dim=1)

    # logits_flat: (batch * max_len, num_classes)
    logits_flat = logits.view(-1, logits.size(-1)) ## -1 means inferred from other dimensions
    #print (logits_flat)
    # log_probs_flat: (batch * max_len, num_classes)
    log_probs_flat = functional.log_softmax(logits_flat,dim=1)
    #print (log_probs_flat)
    # target_flat: (batch * max_len, 1)
    target_flat = target.view(-1, 1).long()
    # losses_flat: (batch * max_len, 1)
    #print (target_flat.size(), log_probs_flat.size())
    losses_flat = -torch.gather(log_probs_flat, dim=1, index=target_flat)
    # losses: (batch, max_len)
    losses = losses_flat.view(*target.size())
    # mask: (batch, max_len)
    #print (logits.float().sum())
    losses = losses * mask
    loss = losses.sum() / (length.float().sum() + 1e-8)
    return loss


def save_model(model, name):
    if not os.path.exists('models/'):
        os.makedirs('models/')

    torch.save(model.state_dict(), 'models/{}.bin'.format(name))


def compute_ent_loss(embedding, logits, target, ent_mask, target_mask):
    """
    Compute loss for entities
    :param embedding:
    :param logits:
    :param target:
    :param ent_mask: entity mask = 1 if entity otherwise 0
    :param target_mask:
    :return:
    """
    cosine_loss = nn.CosineSimilarity(dim=-1)

    pred_out = torch.argmax(logits, dim=-1).transpose(0,1)  # B X S

    pred_out = ent_mask * pred_out

    pred_emb = embedding(pred_out)  # B X S X E

    target_ent = target * ent_mask

    target_emb = embedding(target_ent)  # B X S X E

    entity_loss = (1 - cosine_loss(target_emb, pred_emb))* ent_mask.type(torch.Tensor)  # B X S
    entity_loss = entity_loss * target_mask

    length = torch.sum(target_mask, dim=1)

    entity_loss = entity_loss.sum() / (length.float().sum() + 1e-8)

    return entity_loss

def load_model(model, name, gpu=True):
    if gpu:
        model.load_state_dict(torch.load('models/{}.bin'.format(name)))
    else:
        model.load_state_dict(torch.load('models/{}.bin'.format(name), map_location=lambda storage, loc: storage))

    return model