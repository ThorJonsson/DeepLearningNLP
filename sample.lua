-- Thor Jonsson
-- This program provides a table of functions to sample an NLP neural network
require 'cutorch'
local althingismadur = {}

function load()
	dl = require 'dataload/'
	rnn = require 'rnn'

	local path = './char_results/'
	local t7file = 'PTB_lstm_seqlen100hiddensize2x1024trainsize2to17.t7'
        return torch.load(path..t7file)
end

althingismadur.DNN = load()
-- Okay almost there but definitely something off with our lstm! :/
function althingismadur.speak_out(table)
    ivocab = althingismadur.DNN.testset.icharvocab
    sequence = ""
    val = ""
    for i,val in ipairs(table) do
        print(val)
        next_char = ivocab[torch.floor(val)]
        if next_char ~= nil then
            sequence = sequence .. ivocab[torch.floor(val)]
            print(sequence)
        end
    end
    print(sequence)
end


function althingismadur.sample()
-- subiter takes in two arguments, i.e. batchsize and epochsize
-- inputs : seqlen x batchsize [x inputsize] TODO I get batchsize x seqlen
-- targets : seqlen x batchsize [x inputsize] 64 x 32 x 15 
    model = althingismadur.DNN.model
    opt = althingismadur.DNN.opt
    targetmodule = althingismadur.DNN.targetmodule
    trainset = althingismadur.DNN.trainset
    for i, inputs, targets in trainset:subiter(1, opt.testsize) do
        -- forward the target through criterion
        -- Double cast is needed because of convolution
        while i < 100 do
         targets = targetmodule:forward(targets) -- intTensor with dimension 32 x 32 
         -- forward the model to obtain an output batch
         output = model:forward(inputs)
         output_table = torch.totable(output[1])
         althingismadur.speak_out(output_table[1])
        end
    end
end

return althingismadur

