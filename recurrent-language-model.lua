-- package for aquiring directory paths amongst other things...
require 'paths'
-- recurrent neural network library
require 'rnn'
-- library for dataloaders
local dl = require 'dataload'
-- The txt load module 
local txt_load = require 'txt_load.lua'

local train = require 'train.lua'

local opt = require 'options.lua'

if not opt.silent then
   table.print(opt)
end
opt.id = opt.id == '' and ('althingi' .. ':' .. dl.uniqueid()) or opt.id

if opt.cuda then
   require 'cunn'
   cutorch.setDevice(opt.device)
end

--[[ data set ]]--

bidirectional = true -- if false it is unidirectional
-- This is where the magic happens:
-- input: train_on_char (boolean), bidirectional (boolean), bat
-- Returns the respective loaders 
local trainset, validset, testset = txt_load.Althingi(bidirectional,{opt.batchsize,1,1})
print("Char Vocabulary size : "..#trainset.icharvocab) 
local char_vocabsize = #trainset.icharvocab
print("Word Vocabulary size : "..#trainset.iwordvocab) 
print("Train set split into "..opt.batchsize.." sequences of length "..trainset:size())

--local trainset, validset, testset = dl.loadPTB({opt.batchsize,1,1})
--if not opt.silent then 
--   print("Vocabulary size : "..#trainset.icharvocab) 
--   print("Train set split into "..opt.batchsize.." sequences of length "..trainset:size())
--end

--[[ language model ]]--

local lm = nn.Sequential()


-- rnn layers
local stepmodule = nn.Sequential() -- applied at each time-step
local inputsize = opt.hiddensize[1]
for i,hiddensize in ipairs(opt.hiddensize) do 
    local rnn

    if opt.gru then -- Gated Recurrent Units
        rnn = nn.GRU(inputsize, hiddensize, nil, opt.dropout/2)
    elseif opt.lstm then -- Long Short Term Memory units
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
local lookup = nn.LookupTable(#trainset.icharvocab, opt.hiddensize[1])
lookup.maxnormout = -1 -- prevent weird maxnormout behaviour
lm:insert(lookup,1) -- input is seqlen x batchsize

-- output layer
softmax = nn.Sequential()
softmax:add(nn.Linear(inputsize, #trainset.icharvocab))
softmax:add(nn.LogSoftMax())
-- encapsulate stepmodule into a Sequencer
lm:add(nn.Sequencer(softmax))

-- remember previous state between batches
lm:remember((opt.lstm or opt.gru) and 'both' or 'eval')

if not opt.silent then
    print"Language Model:"
    print(lm)
end

if opt.uniform > 0 then
    for k,param in ipairs(lm:parameters()) do
        param:uniform(-opt.uniform, opt.uniform)
    end
end

--[[ loss function ]]--

local crit = nn.ClassNLLCriterion()

-- target is also seqlen x batchsize.
local targetmodule = nn.SplitTable(1)
if opt.cuda then
    targetmodule = nn.Sequential()
    :add(nn.Convert())
    :add(targetmodule)
end

local criterion = nn.SequencerCriterion(crit)

--[[ CUDA ]]--

if opt.cuda then
    lm:cuda()
    criterion:cuda()
    targetmodule:cuda()
end

--[[ experiment log ]]--

opt.lr = opt.startlr
opt.trainsize = opt.trainsize == -1 and trainset:size() or opt.trainsize
opt.validsize = opt.validsize == -1 and validset:size() or opt.validsize
-- is saved to file every time a new validation minima is found
local xplog = {}
xplog.opt = opt -- save all hyper-parameters and such
xplog.dataset = 'Althingi'
--xplog.vocab = trainset.vocab
xplog.trainset = trainset
xplog.validset = validset
xplog.testset = testset
-- will only serialize params
xplog.model = nn.Serial(lm)
xplog.model:mediumSerial()
xplog.criterion = criterion
xplog.targetmodule = targetmodule
-- keep a log of NLL for each epoch
xplog.trainppl = {}
xplog.valppl = {}
-- will be used for early-stopping
xplog.minvalppl = 99999999
xplog.epoch = 1
xplog = train.run(xplog)
