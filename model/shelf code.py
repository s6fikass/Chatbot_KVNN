

class KVEncoderRNN(nn.Module):
    def __init__(self, input1_size,input2_size, hidden_size, n_layers=1, dropout=0.1):
        super(KVEncoderRNN, self).__init__()

        self.input1_size = input1_size
        self.input2_size = input2_size
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.dropout = dropout

        self.embedding1 = nn.Embedding(input1_size, hidden_size)
        self.embedding2 = nn.Embedding(input2_size, hidden_size)
        self.lstm = nn.LSTM(input1_size, hidden_size)

    def forward(self, input_seqs, input_kb_seqs, hidden=None):

        embedded = self.embedding1(input_seqs)
        kb_embedded = self.embedding2(input_kb_seqs)
        output, hidden = self.lstm(embedded, hidden)

        return output, hidden, kb_embedded

class KbAttn(nn.Module):
    def __init__(self, hidden_size):
        super(KbAttn, self).__init__()

        self.hidden_size = hidden_size
        self.attn = nn.Linear(self.hidden_size * 2, hidden_size)
        self.v = nn.Parameter(torch.FloatTensor(1, hidden_size))

    def forward(self, hidden, k_embedding):

        max_len = 431

        this_batch_size = hidden.size(1)

        print("max_len",max_len)
        print("this_batch_size", this_batch_size)

        # Create variable to store attention energies
        attn_energies = Variable(torch.zeros(this_batch_size, max_len))  # B x S

        # if USE_CUDA:
        #     attn_energies = attn_energies.cuda()

        # For each batch of encoder outputs
        for b in range(this_batch_size):
            # Calculate energy for each encoder output
            for i in range(max_len):

                attn_energies[b, i] = self.score(hidden[:, b], encoder_outputs[i, b].unsqueeze(0))

        # Normalize energies to weights in range 0 to 1, resize to 1 x B x S
        return attn_energies

    def score(self, hidden, encoder_output):
        print(torch.cat((hidden, encoder_output), 1).shape)
        energy = self.attn(torch.cat((hidden, encoder_output), 1))
        energy = self.v.dot(energy)
        return energy


class KVAttnDecoderRNN(nn.Module):
    def __init__(self, attn_model, hidden_size, output_size, n_layers=1, dropout=0.1):
        super(KVAttnDecoderRNN, self).__init__()

        # Keep for reference
        self.attn_model = attn_model
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers
        self.dropout = dropout

        # Define layers

        self.embedding = nn.Embedding(output_size, hidden_size)
        self.embedding_kb = nn.Embedding(output_size, hidden_size)
        self.embedding_dropout = nn.Dropout(dropout)
        self.gru = nn.GRU(hidden_size, hidden_size, n_layers, dropout=dropout)
        self.lstm = nn.LSTM(hidden_size, hidden_size)
        self.concat = nn.Linear(hidden_size * 2, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)

        # Choose attention model
        if attn_model != 'none':
            self.attn = Attn(attn_model, hidden_size)
            self.kbattn = KbAttn(hidden_size)

    def reshapeKb(self, kb_embeding):
        embedding = torch.sum(kb_embeding, dim=2)
        return embedding.reshape(10, self.hidden_size, 431)

    def kbLogits(self, kb, batch_size, pad_length):
        # Create variable to store attention energies of 0 for non kb entities
        v = Variable(torch.zeros(batch_size, pad_length, 1523))  # B x S x Vocab_size
        print(kb.shape)
        attn_energies = kb.reshape(batch_size, pad_length, 431)
        tensor = torch.cat([v, attn_energies], axis=2)
        print(tensor.shape)
        return tensor

    def forward(self, input_seq, kb_inputs, last_context, last_hidden, encoder_outputs):
        # Note: we run this one step at a time (in order to do teacher forcing)

        # Get the embedding of the current input word (last output word)
        batch_size = input_seq.size(0)
        print('[decoder] input_seq', input_seq.size())  # batch_size x 1

        # Decoder Embedding
        embedded = self.embedding(input_seq)
        embedded = self.embedding_dropout(embedded)
        embedded = embedded.view(1, batch_size, self.hidden_size)  # S=1 x B x N
        print('[decoder] word_embedded', embedded.size())

        # Get current hidden state from input word and last hidden state

        print('[decoder] last_hidden', last_hidden.size())
        rnn_output, hidden = self.gru(embedded, last_hidden)
        print('[decoder] rnn_output', rnn_output.size())

        # Calculate attention from current RNN state and all encoder outputs;
        # apply to encoder outputs to get weighted average
        attn_weights = self.attn(rnn_output, encoder_outputs)
        print('[decoder] attn_weights', attn_weights.size())
        print('[decoder] encoder_outputs', encoder_outputs.size())

        context = attn_weights.bmm(encoder_outputs.transpose(0, 1))  # B x S=1 x N

        print('[decoder] context', context.size())

        embedded2 = self.embedding_kb(kb_inputs)
        print('[KB] word_embedded', embedded2.size())
        embedded2 = self.reshapeKb(embedded2)
        print('[KB] reshaped_word_embedded', embedded2.size())

        print("calculating W1 [ kj, ~hi] ")
        kb_attn = self.kbattn(embedded2, last_hidden)

        print(kb_attn)

        kb_attn = self.kbLogits(embedded2, batch_size, encoder_outputs.size(0))

        # Attentional vector using the RNN hidden state and context vector
        # concatenated together (Luong eq. 5)
        rnn_output = rnn_output.squeeze(0)  # S=1 x B x N -> B x N
        context = context.squeeze(1)  # B x S=1 x N -> B x N
        #         print('[decoder] rnn_output', rnn_output.size())
        #         print('[decoder] context', context.size())
        concat_input = torch.cat((rnn_output, context), 1)
        concat_output = F.tanh(self.concat(concat_input))

        # Finally predict next token (Luong eq. 6)
        #         output = F.log_softmax(self.out(concat_output))
        output = self.out(concat_output)

        # Return final output, hidden state, and attention weights (for visualization)
        return output, context, hidden, attn_weights, kb_attn


def train_kb(args, input_batches, target_batches, kb_batch, encoder, decoder, encoder_optimizer,
          decoder_optimizer, criterion,batch_size,target_lengths, clip = 50.0):

    # Zero gradients of both optimizers
    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()
    loss = 0  # Added onto for each word

    # Run words through encoder
    encoder_outputs, encoder_hidden = encoder(input_batches, None)

    # Prepare input and output variables

    decoder_input = Variable(torch.LongTensor([[SOS_token] * batch_size])).transpose(0, 1)
    #     print('decoder_input', decoder_input.size())
    decoder_context = encoder_outputs[-1]
    decoder_hidden = encoder_hidden  # Use last hidden state from encoder to start decoder

    max_target_length = target_batches.shape[0]
    all_decoder_outputs = Variable(torch.zeros(max_target_length, batch_size, decoder.output_size))

    # Move new Variables to CUDA
    if USE_CUDA:
        decoder_input = decoder_input.cuda()
        decoder_context = decoder_context.cuda()
        all_decoder_outputs = all_decoder_outputs.cuda()

    # Choose whether to use teacher forcing
    use_teacher_forcing = random.random() < teacher_forcing_ratio


    if True:
        # Run through decoder one time step at a time
        for t in range(max_target_length):

            decoder_output, decoder_context, decoder_hidden, decoder_attn, kb_attn = decoder(
                decoder_input, kb_batch, decoder_context, decoder_hidden, encoder_outputs
            )
            all_decoder_outputs[t] = decoder_output

            decoder_input = target_batches[t]
    print(all_decoder_outputs.shape)
    loss = masked_cross_entropy(
            all_decoder_outputs.transpose(0, 1).contiguous(),  # seq x batch -> batch x seq
            target_batches.transpose(0, 1).contiguous(),  # seq x batch -> batch x seq
        target_lengths
        )

    loss.backward()

    # Clip gradient norm
    ec = torch.nn.utils.clip_grad_norm_(encoder.parameters(), clip)
    dc = torch.nn.utils.clip_grad_norm_(decoder.parameters(), clip)

    # Update parameters with optimizers
    encoder_optimizer.step()
    decoder_optimizer.step()
    return loss.item(), ec, dc



 # Initialize models
    if args.intent:
        encoder = EncoderRNN(textdata.getVocabularySize(), hidden_size, n_layers, dropout=dropout)
        decoder = LuongAttnDecoderRNN(attn_model, hidden_size, textdata.getVocabularySize(), n_layers, dropout=dropout
                                      , use_cuda=args.cuda)
    else:
        encoder = EncoderRNN(textdata.getVocabularySize(), hidden_size, n_layers, dropout=dropout)
        decoder = LuongAttnDecoderRNN(attn_model, hidden_size, textdata.getVocabularySize(), n_layers, dropout=dropout
                                      , use_cuda=args.cuda)


 if args.loadFilename:
        checkpoint = torch.load(args.loadFilename)
        encoder.load_state_dict(checkpoint['enc'])
        decoder.load_state_dict(checkpoint['dec'])

    # Initialize optimizers and criterion
    encoder_optimizer = optim.Adam(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.Adam(decoder.parameters(), lr=learning_rate * decoder_learning_ratio)
    criterion = nn.CrossEntropyLoss()

    if args.loadFilename:
        encoder_optimizer.load_state_dict(checkpoint['enc_opt'])
        decoder_optimizer.load_state_dict(checkpoint['dec_opt'])

    # Move models to GPU
    if args.cuda:
        encoder=encoder.cuda()
        decoder=decoder.cuda()


