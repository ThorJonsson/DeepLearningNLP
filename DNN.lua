-- 29/07/16
--
local DNN = {}

function DNN.build(opt,M)
    local lm = nn.Sequential()
    -- rnn layers
    local stepmodule = nn.Sequential() -- applied at each time-step
    local inputsize = opt.hiddensize[1]
    for i,hiddensize in ipairs(opt.hiddensize) do 
        local rnn
        if opt.lstm then -- Long Short Term Memory units
            require 'nngraph'
            nn.FastLSTM.usenngraph = true -- faster
            rnn = nn.FastLSTM(inputsize, hiddensize)
            stepmodule:add(rnn)
            stepmodule:add(nn.Dropout(opt.dropout))
        elseif opt.blstm then 
            rnn = nn.Sequencer(nn.FastLSTM(inputsize, hiddensize))--
            lm:add(rnn)
            lm:add(nn.Sequencer(nn.Dropout(opt.dropout)))
        end 
        inputsize = hiddensize
    end

    if opt.blstm then   
        local bwd = lm:clone()
        bwd:reset()
        bwd:remember('neither')
        local bwd_lstm = nn.BiSequencerLM(lm, bwd)

        lm = nn.Sequential()
        lm:add(bwd_lstm)
        inputsize = inputsize*2
    end

    if opt.blstm then
        lm:insert(nn.SplitTable(1),1) -- tensor to table of tensors TODO WHY???
    else
        lm:insert(nn.SplitTable(1),1)
    end
    if opt.dropout > 0 and not opt.gru then  -- gru has a dropout option
        lm:insert(nn.Dropout(opt.dropout),1)
    end
    -- input layer (i.e. word embedding space)
    local lookup = nn.LookupTable(M, opt.hiddensize[1])
    lookup.maxnormout = -1 -- prevent weird maxnormout behaviour
    lm:insert(lookup,1) -- input is seqlen x batchsize

    -- output layer
    softmax = nn.Sequential()
    softmax:add(nn.Linear(inputsize,M))
    softmax:add(nn.LogSoftMax())
    -- encapsulate stepmodule into a Sequencer
    lm:add(nn.Sequencer(softmax))

    -- remember previous state between batches
    lm:remember((opt.lstm or opt.gru) and 'both' or 'eval')

    return lm
end

return DNN
