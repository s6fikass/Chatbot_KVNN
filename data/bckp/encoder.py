import torch
import torch.nn as nn
from torch.autograd import Variable

class RNNModel(nn.Module):
    def __init__(self, rnn_type, ntoken, ninp, nhid, nlayers, dropout=0.5, dropouth=0.5, dropouti=0.5, dropoute=0.1,
                 wdrop=0, tie_weights=False):
        super(RNNModel, self).__init__()
        self.encoder = nn.Embedding(ntoken, ninp)
        self.rnns = [
            torch.nn.LSTM(ninp if l == 0 else nhid, nhid if l != nlayers - 1 else (ninp if tie_weights else nhid), 1,
                          dropout=0) for l in range(nlayers)]
        for rnn in self.rnns:
            rnn.linear = WeightDrop(rnn.linear, ['weight'], dropout=wdrop)

        self.rnns = torch.nn.ModuleList(self.rnns)
        self.attn_fc = torch.nn.Linear(ninp, 1)
        self.decoder = nn.Linear(nhid, ntoken)

        self.init_weights()


    def attention(self, rnn_out, state):
        state = state.squeeze(0).unsqueeze(2)
        weights = torch.bmm(rnn_out, state)
        weights = torch.nn.functional.softmax(weights.squeeze(2)).unsqueeze(2)
        # (batch, cell_size, seq_len) * (batch, seq_len, 1) = (batch, cell_size, 1)
        return torch.bmm(torch.transpose(rnn_out, 1, 2), weights).squeeze(2)


    def forward(self, input, hidden, return_h=False, decoder=False, encoder_outputs=None):
        emb = embedded_dropout(self.encoder, input, dropout=self.dropoute if self.training else 0)
        emb = self.lockdrop(emb, self.dropouti)

        raw_output = emb
        new_hidden = []
        raw_outputs = []
        outputs = []
        for l, rnn in enumerate(self.rnns):
            current_input = raw_output
            raw_output, new_h = rnn(raw_output, hidden[l])
            new_hidden.append(new_h)
            raw_outputs.append(raw_output)
            if l != self.nlayers - 1:
                raw_output = self.lockdrop(raw_output, self.dropouth)
                outputs.append(raw_output)
        hidden = new_hidden

        output = self.lockdrop(raw_output, self.dropout)
        outputs.append(output)

        attn_out = self.attention(raw_output, new_h[0])

        decoded = self.decoder(attn_out)
        return decoded