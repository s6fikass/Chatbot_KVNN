
import torch
import torch.nn as nn
from torch import optim
from torch.nn import functional
from torch.autograd import Variable
import numpy as np
import random
from sklearn.metrics.pairwise import cosine_distances


SOS_token = 1
teacher_forcing_ratio = 0.5

def isEntity(data, targetword, kb):

    if len(kb)>1:
        for k in kb:
            if targetword == k[0] and targetword != 0:
                return True
            if targetword == k[2] and targetword != 0:
                return True
    else:
        return False

    return True

def sequence_mask(sequence_length, max_len=None,USE_CUDA=False):
    if max_len is None:
        max_len = sequence_length.data.max()
    batch_size = sequence_length.size(0)
    seq_range = torch.range(0, max_len - 1).long()
    seq_range_expand = seq_range.unsqueeze(0).expand(batch_size, max_len)

    seq_range_expand = Variable(seq_range_expand)

    if USE_CUDA:
        seq_range_expand = seq_range_expand.cuda()

    seq_length_expand = (sequence_length.unsqueeze(1)
                         .expand_as(seq_range_expand))
    return seq_range_expand < seq_length_expand


def masked_cross_entropy(logits, target, target_lengths,USE_CUDA=False):
    length = Variable(torch.LongTensor(target_lengths))

    if USE_CUDA:
        length = length.cuda()

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

    # logits_flat: (batch * max_len, num_classes)
    logits_flat = logits.view(-1, logits.size(-1))
    # log_probs_flat: (batch * max_len, num_classes)
    log_probs_flat = functional.log_softmax(logits_flat, dim=1)
    # target_flat: (batch * max_len, 1)
    target_flat = target.view(-1, 1)
    # losses_flat: (batch * max_len, 1)
    losses_flat = -torch.gather(log_probs_flat, dim=1, index=target_flat)
    # losses: (batch, max_len)
    losses = losses_flat.view(*target.size())
    # mask: (batch, max_len)
    mask = sequence_mask(sequence_length=length, max_len=target.size(1))

    losses = losses * mask.float()
    loss = losses.sum() / length.float().sum()
    return loss


def entity_similarity_loss(data, word, predicted):
    true_seq = data.id2word[word]
    prediceted_seq = data.id2word[predicted]

    true_vec = np.zeros(300)
    predicted_vec = np.zeros(300)

    for w in true_seq.split(" "):
        true_vec = np.add(true_vec,data.word_to_embedding_dict[w])

    true_vec = np.divide(true_vec,len(true_seq.split(" "))).reshape(1, -1)

    for w in prediceted_seq.split(" "):
        predicted_vec = np.add(predicted_vec, data.word_to_embedding_dict[w])

    predicted_vec = np.divide(predicted_vec, len(prediceted_seq.split(" "))).reshape(1, -1)

    return cosine_distances(true_vec, predicted_vec)[0, 0]


def train(args, input_batches, target_batches, encoder, decoder, encoder_optimizer,
          decoder_optimizer, criterion,batch_size, input_lengths, target_lengths, clip = 50.0,kb=[],intent_batch=None):

    # Zero gradients of both optimizers
    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()
    loss = 0  # Added onto for each word

    # Run words through encoder
    encoder_outputs, encoder_hidden = encoder(input_batches, input_lengths)

    # Prepare input and output variables

    decoder_input = Variable(torch.LongTensor([[SOS_token] * batch_size])).transpose(0, 1)
    #     print('decoder_input', decoder_input.size())
    decoder_context = encoder_outputs[-1]
    decoder_hidden = encoder_hidden  # Use last hidden state from encoder to start decoder

    intent_output = torch.LongTensor(intent_batch)

    max_target_length = target_batches.shape[0]
    all_decoder_outputs = Variable(torch.zeros(max_target_length, batch_size, decoder.output_size))

    # Move new Variables to CUDA
    if args.cuda:
        decoder_input = decoder_input.cuda()
        # decoder_context = decoder_context.cuda()
        all_decoder_outputs = all_decoder_outputs.cuda()
        intent_output = intent_output.cuda()

    # Choose whether to use teacher forcing
    use_teacher_forcing = random.random() < teacher_forcing_ratio

    entity_additional_loss = 0
    entity_loss_cof = 0.8

    # TODO: Get targets working
    if True:
        # Run through decoder one time step at a time
        for t in range(max_target_length):

            if args.intent and t == 0:
                decoder_output, decoder_context, decoder_hidden, decoder_attn, intent_score = decoder(
                    decoder_input, decoder_context, decoder_hidden, encoder_outputs, intent_batch=True
                )
            else:
                decoder_output, decoder_context, decoder_hidden, decoder_attn,_ = decoder(
                    decoder_input, decoder_context, decoder_hidden, encoder_outputs
                )

            all_decoder_outputs[t] = decoder_output
            topv, topi = decoder_output.data.topk(1)
            topi=topi.squeeze(1)

            if args.glove:
                for idx,word in enumerate(decoder_input):
                    if len(kb[idx])>1:
                        if isEntity(args.data, word.item(), kb[idx]):
                            entity_additional_loss += entity_similarity_loss(args.data, word.item(), topi[idx].item())

            decoder_input = target_batches[t]

    loss = masked_cross_entropy(
            all_decoder_outputs.transpose(0, 1).contiguous(),  # seq x batch -> batch x seq
            target_batches.transpose(0, 1).contiguous(),  # seq x batch -> batch x seq
        target_lengths
        )

    if args.intent:
        loss_function_2 = nn.CrossEntropyLoss()
        intent_loss = loss_function_2(intent_score, intent_output)
        loss = loss.add(2*intent_loss.item())

    loss = loss.add(entity_additional_loss*entity_loss_cof)
    loss.backward()

    # Clip gradient norm
    ec = torch.nn.utils.clip_grad_norm_(encoder.parameters(), clip)
    dc = torch.nn.utils.clip_grad_norm_(decoder.parameters(), clip)

    # Update parameters with optimizers
    encoder_optimizer.step()
    decoder_optimizer.step()
    return loss.item(), ec, dc

