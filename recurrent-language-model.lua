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

bidirectional = false -- if false it is unidirectional
-- This is where the magic happens:
-- input: train_on_char (boolean), bidirectional (boolean), bat
-- Returns the respective loaders 
local trainset, validset, testset = txt_load.PTB(bidirectional,{opt.batchsize,1,1},opt.datapath)
print("Char Vocabulary size : "..#trainset.ivocab) 
local char_vocabsize = #trainset.ivocab
print("Word Vocabulary size : "..#trainset.ivocab) 
print("Train set split into "..opt.batchsize.." sequences of length "..trainset:size())

--local trainset, validset, testset = dl.loadPTB({opt.batchsize,1,1})
--if not opt.silent then 
--   print("Vocabulary size : "..#trainset.icharvocab) 
--   print("Train set split into "..opt.batchsize.." sequences of length "..trainset:size())
--end

--[[ language model ]]--
local DNN = require 'DNN.lua'
local lm = DNN.build(opt,#trainset.ivocab)

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

-- target is also seqlen x batchsize.TODO: explain
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
xplog.dataset = 'PTB'
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
