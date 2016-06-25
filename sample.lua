-- Thor Jonsson
-- This program provides a table of functions to sample an NLP neural network

local sampler = {}

function sampler.load()
	dl = require 'dataload/'
	rnn = require 'rnn'
	local path = './char_results/'
	local t7file = 'althingi_blstm_seqlen100hiddensize1x200trainsize2to16.t7'
    local althingismadur = torch.load(path..t7file)
	return althingismadur
end

return sampler






---- Loads the checkpoint
--function load()
--    model = xp:model()
--    print(torch.type(model))
--  
--end
--
---- Asks user to give input to seed
----function ask4input()
----  io.write('Give the computer an input to correct: \n')
----  io.flush()
----  local answer = io.read()
----  -- split the answer into words and put it in a table
----  return split_string(answer)
----end
--
----[[
----This file samples characters from a trained model
----Code
----[[
--This file samples characters from a trained model
--Code is based on implementation in Andrej Karpathy's https://github.com/karpathy/char-rnn
--which was in turn based on implementation in 
--https://github.com/oxford-cs-ml-2015/practical6
--]]--
--
--require 'nn'
--require 'rnn'
--require 'sys'
--require 'os'
--require 'torch'
--require 'paths'
--require 'nngraph'
--
--cmd = torch.CmdLine()
--cmd:text()
--cmd:text('Sample from a character-level language model')
--cmd:text()
--cmd:text('Options')
--cmd:argument('-xplog','experiment log to use for sampling (generated by train.lua)')
--cmd:option('-back', 'cpu', 'cpu|cuda|cl')
--cmd:option('-len', 2000, 'number of characters to sample')
--cmd:option('-temp', 1, 'temperature of sampling')
--cmd:option('-device', 1, 'which GPU device to use')
--cmd:text()
--
--local opt = cmd:parse(arg)
--
--local xplog = torch.load("./blstm_2-high_2-lstm_2/fortran:1459010874:1.dat")
--local model = xplog:model()
--local net = model.module -- nn.Serial(module) -> module
--local vocab = xplog.vocab
--local ivocab = {}
--for char, i in pairs(vocab) do
--  ivocab[i] = char
--end
--
--print('net', net)
--
--net:float()
--
--local q
--local function sampleSoftMax(log_q, t)
--  -- eg see https://en.wikipedia.org/wiki/Softmax_function, section 'Reinforcement learning'
--  q = q or log_q.new()
--  q:exp(log_q)
--  q:mul(1/t)
--  local sum_q = q:sum()
--  q:div(sum_q)
--  local sample = torch.multinomial(q, 1)
--  return sample
--end
--
---- seed with '\n' for now. a bit too much prior knowledge introduced by doing this, but
---- gets it working for now....
--local newLine = '\n'
--local prevChar = vocab[newLine]
--
--net:evaluate()
--net:remember('both')
--
--local sample = {}
--local input = torch.LongTensor(1,1)
--for i=1,opt.len do
--  input[{1,1}] = prevChar
--  local output = net:forward(input)[1]
--  local thisChar = sampleSoftMax(output[1], opt.temp)
--  local sampleChar = ivocab[thisChar[1]]
--  table.insert(sample, sampleChar)
--  prevChar = thisChar[1]
--end
--print(table.concat(sample,''))
--

