-- Thor Jonsson 26/05'16
-- Collection of functions which load various text datasets 

-- Dependencies
local utf8 = require 'lua-utf8'
local txt_load_util = require 'txt_load_util.lua'
--local  = require 
-- We initialize an empty table of text dataload methods.
local txt_load = {}

-- Loads Althingi train, valid, test sets
-- inputs 
--      char (boolean) train on characters or words
--      bidirectional (boolean) train bidirectionally or not
--      batchsize (table) contains batchsizes for train, test and valid respectively
function txt_load.Althingi(train_on_char,is_bidirectional,batchsize)
    -- 1. arguments and defaults
    -- the size of the batch is fixed for SequenceLoaders
    batchsize = torch.type(batchsize) == 'table' and batchsize or {batchsize, batchsize, batchsize}
    assert(torch.type(batchsize[1]) == 'number')
    -- path to directory containing Penn Tree Bank dataset on disk
    -- the dir contains train.txt, valid.txt and test.txt
    datapath = '/home/thj92/DeepLearningNLP/Data/'

    -- 2. load raw data, convert to tensor

    local file = require('pl.file')
    local stringx = require('pl.stringx')

    local loaders = {}
    for i,whichset in ipairs{'train', 'valid', 'test'} do
        -- download the file if necessary
        local filename = 'althingi.'..whichset..'.txt'
        local filepath = paths.concat(datapath, filename)
        local text = file.read(filepath)
        text = stringx.replace(text, '\n', '<eos>')
        local tokens = {}
        -- First we do the characters
        --if ischartokens then TODO ADD support to choose between word and char
        -- TODO can we use an online algorithm to do this?
        for c in utf8.gmatch(text,'.') do -- splits into characters
            if c ~= '_' then
                table.insert(tokens, c)
            end  
        end
        -- We have collected all the char tokens
        -- Now we build a vocabulary for those tokens.
        if whichset == 'train' and ischartokens then
            charvocab, icharvocab, charfreq = txt_load_util.buildVocab(tokens,ischartokens)
        end
        -- outputs a tensor with dimensions 1 x #vocab
        local tensor = txt_load_util.text2tensor(tokens, charvocab)
        local loader = dl.SequenceLoader(tensor, batchsize[i],bidirectional)
        -- reset tokens to capture the words
        tokens = {} 
        tokens = stringx.split(text) -- splits into words
        if whichset == 'train' then
            wordvocab, iwordvocab, wordfreq, maxwordlen = txt_load_util.buildVocab(tokens, false)
        end
        -- outputs a tensor with dimensions 1 x #vocab
        --local tensor = dl.text2tensor(tokens, wordvocab) TODO but unnecessary at this point
        -- 3. encapsulate into SequenceLoader
        --local loader = dl.SequenceLoader(tensor, batchsize[i],bidirectional)

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

return txt_load
