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
        local text = txt_load_util.getAlthingi(txt_set)
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

return txt_load
