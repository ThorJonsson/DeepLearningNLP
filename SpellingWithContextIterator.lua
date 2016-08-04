-- We want to be able to take a string of sentences seperated by <eos> and use each of them
-- independently to learn how to fill in a word given the context. What we want to accomplish
-- is a function which maps 
--
-- Step 1:
-- break string on <eos>:
-- Feed sentence 1 as it is correctly as an input
-- x_1 x_2 ... x_n
-- Then make a batch such that each word occurs once
-- <unk> x_2 ... x_n
-- x_1 <unk> ... x_n
-- .
-- .
-- .
-- x_1 x_2 ... <unk>
--
-- for each try to predict the correct value for <unk>
--
-- the corresponding output should be
--
-- y_1 x_2 ... x_n
-- x_1 y_2 ... x_n
-- .
-- .
-- .
-- x_1 x_2 ... y_n
--
-- and the corresponding output should be
--
-- x_1 x_2 ... x_n
-- x_1 x_2 ... x_n
-- .
-- .
-- .
-- x_1 x_2 ... x_n
--
-- Make a table that associates outputs of the neural network with the corresponding targets
-- With a certain probability replace the top ten most common outputs with the input. During the training regime to obtain corrections.
--
-- Suppose text is all our testdata
local txt_load_util = require 'txt_load_util.lua'
local text = txt_load_util.getAlthingi('test')
local utf8 = require 'lua-utf8'
-- We add this function to txt_load_util
function string:split(scale) -- scale: char, word, snt, blob (TBI),txt
    local N = utf8.len(self)
    local token -- token depends on the scale
    local split_start = 1
    -- How we choose splitend depends on the scale
    -- If scale is char then it's the next char position
    -- If scale is word then it's the position as determined by find with token = ' '
    -- If scale is snt then it's the position as determined by snt with token = '<eos>'
    -- If scale is blob then it's the position as determined by some attention mechanism TBI
    local split_end
    local output = {}
    if scale == 'char' then
        for i=1,N do
            output[i] = utf8.sub(self,split_start, split_start)
            split_start = split_start + 1
        end
    else 

        if scale == 'word' then
            token = ' '
        elseif scale == 'snt' then
            token = '<eos>'
        end
        -- Problem: utf8.find returns nil if there's no remaining ' ' left.
        -- Check if split_end = nil, if so grab set split_end = N
        while split_start <= N do -- TODO use tokens to eliminate repeated spaces
            _, split_end = utf8.find(self,token,split_start)
            if split_end == nil then split_end = N end
            local str = utf8.sub(self,split_start, split_end)
            if str ~= token then 
                table.insert(output, word)
            end
            split_start = split_end + 1
        end
    end

    return output
end
-- Example of usage:
--th> txt = 'Ég fór út að labba.<eos>Hvernig var veðrið?<eos>Hvar er mamma?<eos>Hún var skelfingu lostin!<eos>'
--                                                                      [0.0000s]
--th> txt:split('<eos>')
--{
--  1 : "Ég fór út að labba."
--  2 : "Hvernig var veðrið?"
--  3 : "Hvar er mamma?"
--  4 : "Hún var skelfingu lostin!"
--}
-- For each sentence we now turn it into the problem to solve
-- For example for sentence 1
-- <unk> fór út að labba.
-- Ég <unk> út að labba.
-- Ég fór <unk> að labba.
-- Ég fór út <unk> labba.
-- Ég fór út að <unk>
function string:test()
    local txt = 'Éfór út að labba.<eos>Hvernig var veðrið?<eos>Hvar er mamma?<eos>Hún var skelfingu lostin!<eos>'
-- splittar í setningar
    local sentences = txt:split('<eos>')
    return sentences
end

-- How to define a sequence_loader class

local sequence_loader = torch.class('sequence_loader')

function sequence_loader:__init(sequence, batchsize, bidirectional)
   assert(torch.isTensor(sequence))
   assert(torch.type(batchsize) == 'number')
   -- sequence is a tensor where the first dimension indexes time
   self.batchsize = batchsize
   self.bidirectional = bidirectional 
   local seqlen = sequence:size(1)
   local size = sequence:size():totable()
   table.remove(size, 1)
   assert(#size == sequence:dim() - 1)
   self.data = sequence.new()
   -- note that some data will be lost
   -- Number of batches
   local seqlen2 = torch.floor(seqlen / batchsize)
   -- seqlen2 x batchsize
   self.data = sequence:sub(1,seqlen2*batchsize):view(batchsize, seqlen2):t():contiguous()
end

A = sequence_loader(torch.rand(5),5,false) -- works but is meaningless
-- input 1 : sequence in tensor form
-- input 2 : batchsize - i.e. the size of each batch
-- input 3 : bidirectional - true or false
-- To prepare input 1:

function txt_load_util.get_raw_data(txt_set,datapath)
    -- Dependencies
    local file = require('pl.file')
    local stringx = require('pl.stringx')
    -- path to directory containing Althingi dataset on disk
    -- This is the current default if no argument given
    -- the dir contains train.txt, valid.txt and test.txt
    datapath = datapath or '/home/thj92/DeepLearningNLP/Data/'
    -- 2. load raw data,
    local filename = 'althingi.'..txt_set..'.txt'
    local filepath = paths.concat(datapath, filename)
    local text = file.read(filepath)
    text = stringx.replace(text, '\n', '<eos>')
    return text
end

local text = txt_load_util.get_raw_data(txt_set, '/home/thj92/DeepLearningNLP/Data/')

-- tokens contain all the characters from the whole sequence representing our document.
-- We are going to build a table containing all the different unique tokens
function txt_load_util.buildVocab(tokens)
    local vocab = {}
	local ivocab = {}
	local counter = 1
    -- Store each character as they appear in tokens
    for i=1,#tokens do
        local char = tokens[i]
		if vocab[char] == nil then
			ivocab[counter] = char
			vocab[char] = counter
			counter = counter + 1
		end 
	end
	return vocab, ivocab
end

local charvocab, icharvocab = txt_load_util.buildVocab(tokens)

function txt_load_util.text2tensor(tokens,vocab)
    -- Build a tensor with a length which corresponds to size of vocabulary
    -- Each element in the vocab will receive an entry in this one-dimensional tensor
    local tensor = torch.IntTensor(#tokens):fill(0)
    -- Each number receives word id which corresponds to its frequency
    for i, char in ipairs(tokens) do
        tensor[i] = vocab[char]
    end
    return tensor
end

local tensor = txt_load_util.text2tensor(tokens, charvocab)

local loader = sequence_loader(tensor,batchsize,true)

-- To disseminate tomorrow!

-- subiter : for iterating over validation and test sets
function DataLoader:subiter(batchsize, epochsize, ...)
   batchsize = batchsize or 32
   local dots = {...}
   local size = self:size()
   epochsize = epochsize or -1 
   epochsize = epochsize > 0 and epochsize or self:size()
   self._start = self._start or 1
   local nsampled = 0
   local stop
   
   local inputs, targets
   
   -- build iterator
   return function()
      if nsampled >= epochsize then
         return
      end
      
      local bs = math.min(nsampled+batchsize, epochsize) - nsampled
      stop = math.min(self._start + bs - 1, size)
      -- inputs and targets
      local batch = {self:sub(self._start, stop, inputs, targets, unpack(dots))}
      -- allows reuse of inputs and targets buffers for next iteration
      inputs, targets = batch[1], batch[2]
      
      bs = stop - self._start + 1
      nsampled = nsampled + bs
      self._start = self._start + bs
      if self._start > size then
         self._start = 1
      end
      
      self:collectgarbage()
      
      return nsampled, unpack(batch)
   end
end

