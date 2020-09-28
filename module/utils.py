from torch import nn

magic_number = -1.0E+10

def run_lstm(module, H):
    H = nn.utils.rnn.pack_sequence(H, enforce_sorted=False)
    Y, _ = module(H)
    Y, lengths = nn.utils.rnn.pad_packed_sequence(Y, batch_first=True)
    Y = [Y[i, :length] for i, length in enumerate(lengths)]
    return Y
