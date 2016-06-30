--This module contains all the different options/setting for the DNN

    opt = {}
    opt.startlr = 0.05  --'learning rate at t=0')
    opt.minlr = 0.00001  --'minimum learning rate')
    opt.saturate = 400--'epoch at which linear decayed LR will reach minlr')
    opt.momentum = 0.9--'momentum')
    opt.maxnormout = -1--'max l2-norm of each layer\'s output neuron weights')
    opt.cutoff = -1--'max l2-norm of concatenation of all gradParam tensors')
    opt.cuda = false--'use CUDA')
    opt.device = 1--'sets the device (GPU) to use')
    opt.maxepoch = 1000--'maximum number of epochs to run')
    opt.earlystop = 50--'maximum number of epochs to wait to find a better local minima for early-stopping')
    opt.progress = true--'print progress bar')
    opt.silent = false--'don\'t print anything to stdout')
    opt.uniform = 0.1--'initialize parameters using uniform distribution between -uniform and uniform. -1 means default initialization')
    -- rnn layer 
    opt.lstm = true--'use Long Short Term Memory (nn.LSTM instead of nn.Recurrent)')
    opt.blstm = false--'use Long Short Term Memory (nn.LSTM instead of nn.Recurrent)')
    opt.gru = false--'use Gated Recurrent Units (nn.GRU instead of nn.Recurrent)')
    opt.seqlen = 100--'sequence length : back-propagate through time (BPTT) for this many time-steps')
    opt.hiddensize = {200,200} -- 'number of hidden units used at output of each recurrent layer. When more than one is specified--RNN/LSTMs/GRUs are stacked')
    opt.dropout = 0.5--'apply dropout with this probability after each rnn layer. dropout <= 0 disables it.')
    -- data
    opt.batchsize = 2^8--'number of examples per batch')
    opt.trainsize = 2^14--'number of train examples seen between each epoch')
    opt.validsize = 2^13 -- number of valid examples used for early stopping and cross-validation
    opt.savepath = './char_results' --  'path to directory where experiment log (includes model) will be saved')
    opt.id = 'althingi_lstm_seqlen100hiddensize2x200trainsize2to10'--'id string of this experiment (used to name output file) (defaults to a unique id)')

    return opt
