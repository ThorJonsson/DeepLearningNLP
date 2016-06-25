require 'paths'
require 'rnn'
local dl = require 'dataload'
local txt_load = require 'txt_load.lua'

local train = require 'train.lua'
version = 3

--[[ command line arguments ]]--
cmd = torch.CmdLine()
cmd:text()
cmd:text('Train a Language Model on PennTreeBank dataset using RNN or LSTM or GRU')
cmd:text('Example:')
cmd:text('th recurrent-language-model.lua --cuda --device 2 --progress --cutoff 4 --seqlen 10')
cmd:text("th recurrent-language-model.lua --progress --cuda --lstm --seqlen 20 --hiddensize '{200,200}' --batchsize 20 --startlr 1 --cutoff 5 --maxepoch 13 --schedule '{[5]=0.5,[6]=0.25,[7]=0.125,[8]=0.0625,[9]=0.03125,[10]=0.015625,[11]=0.0078125,[12]=0.00390625}'")
cmd:text("th recurrent-language-model.lua --progress --cuda --lstm --seqlen 35 --uniform 0.04 --hiddensize '{1500,1500}' --batchsize 20 --startlr 1 --cutoff 10 --maxepoch 50 --schedule '{[15]=0.87,[16]=0.76,[17]=0.66,[18]=0.54,[19]=0.43,[20]=0.32,[21]=0.21,[22]=0.10}' -dropout 0.65")
cmd:text('Options:')
-- training
cmd:option('--startlr', 0.05, 'learning rate at t=0')
cmd:option('--minlr', 0.00001, 'minimum learning rate')
cmd:option('--saturate', 400, 'epoch at which linear decayed LR will reach minlr')
cmd:option('--schedule', '', 'learning rate schedule. e.g. {[5] = 0.004, [6] = 0.001}')
cmd:option('--momentum', 0.9, 'momentum')
cmd:option('--maxnormout', -1, 'max l2-norm of each layer\'s output neuron weights')
cmd:option('--cutoff', -1, 'max l2-norm of concatenation of all gradParam tensors')
cmd:option('--cuda', false, 'use CUDA')
cmd:option('--device', 1, 'sets the device (GPU) to use')
cmd:option('--maxepoch', 1000, 'maximum number of epochs to run')
cmd:option('--earlystop', 50, 'maximum number of epochs to wait to find a better local minima for early-stopping')
cmd:option('--progress', true, 'print progress bar')
cmd:option('--silent', false, 'don\'t print anything to stdout')
cmd:option('--uniform', 0.1, 'initialize parameters using uniform distribution between -uniform and uniform. -1 means default initialization')
-- rnn layer 
cmd:option('--lstm', false, 'use Long Short Term Memory (nn.LSTM instead of nn.Recurrent)')
cmd:option('--blstm', true, 'use Long Short Term Memory (nn.LSTM instead of nn.Recurrent)')
cmd:option('--gru', false, 'use Gated Recurrent Units (nn.GRU instead of nn.Recurrent)')
cmd:option('--seqlen', 100, 'sequence length : back-propagate through time (BPTT) for this many time-steps')
cmd:option('--hiddensize', '{200}', 'number of hidden units used at output of each recurrent layer. When more than one is specified, RNN/LSTMs/GRUs are stacked')
cmd:option('--dropout', 0.5, 'apply dropout with this probability after each rnn layer. dropout <= 0 disables it.')
-- data
cmd:option('--batchsize', 2^8, 'number of examples per batch')
cmd:option('--trainsize', 2*32768, 'number of train examples seen between each epoch')
cmd:option('--validsize', 2*16384, 'number of valid examples used for early stopping and cross-validation') 
cmd:option('--savepath', './char_results', 'path to directory where experiment log (includes model) will be saved')
cmd:option('--id', 'althingi_blstm_seqlen100hiddensize1x200trainsize2to16', 'id string of this experiment (used to name output file) (defaults to a unique id)')

cmd:text()
local opt = cmd:parse(arg or {})
opt.hiddensize = loadstring(" return "..opt.hiddensize)()
opt.schedule = loadstring(" return "..opt.schedule)()
if not opt.silent then
   table.print(opt)
end
opt.id = opt.id == '' and ('althingi' .. ':' .. dl.uniqueid()) or opt.id

if opt.cuda then
   require 'cunn'
   cutorch.setDevice(opt.device)
end

--[[ data set ]]--

train_on_char = true -- if false it trains on words
bidirectional = true -- if false it is unidirectional
-- This is where the magic happens:
-- input: train_on_char (boolean), bidirectional (boolean), bat
-- Returns the respective loaders 
local trainset, validset, testset = txt_load.Althingi(train_on_char,bidirectional,{opt.batchsize,1,1})
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
