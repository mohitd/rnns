import torch
import torch.nn as nn
import torch.nn.functional as F


class RNN(nn.Module):

    def __init__(self,
                 input_size: int,
                 hidden_size: int,
                 output_size: int,
                 _num_layers: int):
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        # define an embedding layer to map one-hot inputs to a learned dense vector
        self.embedding = nn.Embedding(input_size, input_size)
        self.i2h = nn.Linear(input_size, hidden_size)
        self.h2h = nn.Linear(hidden_size, hidden_size)
        self.h2o = nn.Linear(hidden_size, output_size)

    def forward(self, x: torch.Tensor, h=None):
        # initialize hidden state if none was provided
        if h is None:
            h = torch.zeros(1, self.hidden_size).to(x.device)

        seq_size, _ = x.size()
        out = []

        # run each token through the RNN and collect the outputs
        for t in range(seq_size):
            embedding = self.embedding(x[t])
            h = F.tanh(self.i2h(embedding) + self.h2h(h))
            o = self.h2o(h)
            out.append(o)
        out = torch.stack(out)

        # detach hidden state so we can optimize over it over the sequence
        return out, h.detach()


class CustomLSTM(nn.Module):

    def __init__(self,
                 input_size: int,
                 hidden_size: int,
                 output_size: int,
                 _num_layers: int):
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.embedding = nn.Embedding(input_size, input_size)

        self.W_i = nn.Linear(input_size, hidden_size)
        self.U_i = nn.Linear(hidden_size, hidden_size)

        self.W_f = nn.Linear(input_size, hidden_size)
        self.U_f = nn.Linear(hidden_size, hidden_size)

        self.W_g = nn.Linear(input_size, hidden_size)
        self.U_g = nn.Linear(hidden_size, hidden_size)

        self.W_o = nn.Linear(input_size, hidden_size)
        self.U_o = nn.Linear(hidden_size, hidden_size)

        self.V = nn.Linear(hidden_size, output_size)

    def forward(self, x: torch.Tensor, h=None):
        # initialize hidden state if none was provided
        if h is None:
            h_t = torch.zeros(1, self.hidden_size).to(x.device)
            c_t = torch.zeros(1, self.hidden_size).to(x.device)
            # "hidden state" for an LSTM is comprised of a normal hidden state
            # and a cell state
            # we concatenate them into a single tuple for consistency
            h = (h_t, c_t)

        seq_size, _ = x.size()
        out = []
        # run each token through the RNN and collect the outputs
        for t in range(seq_size):
            x_t = self.embedding(x[t])

            # decompose "hidden" state into the hidden and cell states
            h_prev = h[0]
            c_prev = h[1]

            # compute gates
            i = F.sigmoid(self.W_i(x_t) + self.U_i(h_prev))
            f = F.sigmoid(self.W_f(x_t) + self.U_f(h_prev))
            g = F.tanh(self.W_g(x_t) + self.U_g(h_prev))
            o = F.sigmoid(self.W_o(x_t) + self.U_o(h_prev))

            # compute new cell and hidden states
            c_t = f * c_prev + i * g
            h_t = o * F.tanh(c_t)

            # package them back into a "hidden" state for the next iteration
            h = (h_t, c_t)

            out.append(self.V(h_t))
        out = torch.stack(out)

        # detach hidden state so we can optimize over it over the sequence
        return out, (h[0].detach(), h[1].detach())


class LSTM(nn.Module):

    def __init__(self,
                 input_size: int,
                 hidden_size: int,
                 output_size: int,
                 num_layers: int = 1):
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.embedding = nn.Embedding(input_size, input_size)
        self.rnn = nn.LSTM(input_size=input_size,
                           hidden_size=hidden_size,
                           num_layers=num_layers)
        self.decoder = nn.Linear(hidden_size, output_size)

    def forward(self, x: torch.Tensor, h=None):
        embedding = self.embedding(x)

        # with PyTorch RNNs, we can pass the whole sequence in one go
        output, h = self.rnn(embedding, h)
        output = self.decoder(output)

        # detach hidden state so we can optimize over it over the sequence
        return output, (h[0].detach(), h[1].detach())
