-- Thor Jonsson 26/05'16
-- Collection of functions which load various text datasets 

-- Dependencies
local utf8 = require 'lua-utf8'
local txt_load_util = require 'txt_load_util.lua'
-- We initialize an empty table of text dataload methods.
local txt_load = {}
local dl = require 'dataload'
-- Loads Althingi train, valid, test sets
-- inputs 
--      is_bidirectional (boolean) train bidirectionally or not
--      batchsize (table) contains batchsizes for train, test and valid respectively
function txt_load.Althingi(is_bidirectional,batchsize)
    -- 1. arguments and defaults
    -- the size of the batch is fixed for SequenceLoaders
    batchsize = torch.type(batchsize) == 'table' and batchsize or {batchsize, batchsize, batchsize}
    assert(torch.type(batchsize[1]) == 'number')
    local charvocab = {}
    local icharvocab = {}
    local loaders = {}
    -- There are three different datafiles in dir
    for i,txt_set in ipairs{'train', 'valid', 'test'} do
        local text = txt_load_util.get_raw_data(txt_set, '/home/thj92/DeepLearningNLP/Data/')
        -- Build the list of unique characters in the text (charvocab)
        -- icharvocab is the inverted list
        local tokens = txt_load_util.tokenize(text)
        if txt_set == 'train' then
            charvocab, icharvocab = txt_load_util.buildVocab(tokens)
        end
        -- outputs a tensor with dimensions 1 x #tokens
        local tensor = txt_load_util.text2tensor(tokens, charvocab)
        
        local loader = dl.SequenceLoader(tensor, batchsize[i],is_bidirectional)
        -- Gather word statistics
        -- reset tokens to capture the words
        tokens = {} 
        tokens = stringx.split(text) -- splits into words
        if txt_set == 'train' then
            wordvocab, iwordvocab, wordfreq, maxwordlen = txt_load_util.buildVocab(tokens, false)
        end

        loader.charvocab = charvocab
        loader.icharvocab = icharvocab
        loader.charfreq = charfreq
        loader.wordvocab = wordvocab
        loader.iwordvocab = iwordvocab
        loader.wordfreq = wordfreq
        loader.maxwordlen = maxwordlen
        table.insert(loaders, loader)
    end

    return unpack(loaders)
end

-- Loads Penn Tree Bank train, valid, test sets
function txt_load.PTB(bidirectional,batchsize, datapath, srcurl, vocab, ivocab, wordfreq)
   -- 1. arguments and defaults
   if bidirectional == true then
       print('We have not implemented bidirectional for PTB yet')
   end

   -- the size of the batch is fixed for SequenceLoaders
   batchsize = torch.type(batchsize) == 'table' and batchsize or {batchsize, batchsize, batchsize}
   assert(torch.type(batchsize[1]) == 'number')
   -- path to directory containing Penn Tree Bank dataset on disk
   datapath = datapath or paths.concat(dl.DATA_PATH, 'PennTreeBank')
   -- URL from which to download dataset if not found on disk.
   srcurl = srcurl or 'https://raw.githubusercontent.com/wojzaremba/lstm/master/data/'
   
   if vocab then
      assert(ivocab and wordfreq)
   end

   -- 2. load raw data, convert to tensor
   
   local file = require('pl.file')
   local stringx = require('pl.stringx')
   
   local loaders = {}
   for i,whichset in ipairs{'train', 'valid', 'test'} do
      -- download the file if necessary
      local filename = 'ptb.'..whichset..'.txt'
      local filepath = paths.concat(datapath, filename)
      dl.downloadfile(datapath, srcurl..filename, filepath)
      local text = file.read(filepath)
      text = stringx.replace(text, '\n', '<eos>')
      local tokens = stringx.split(text)
      if whichset == 'train' and not vocab then
         vocab, ivocab, wordfreq = dl.buildVocab(tokens)
      end
      local tensor = dl.text2tensor(tokens, vocab)
      
      -- 3. encapsulate into SequenceLoader
      local loader = dl.SequenceLoader(tensor, batchsize[i])
      loader.vocab = vocab
      loader.ivocab = ivocab
      loader.wordfreq = wordfreq
      table.insert(loaders, loader)
   end
   
   return unpack(loaders)
end

return txt_load
