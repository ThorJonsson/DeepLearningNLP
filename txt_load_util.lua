-- Thor Jonsson 26/05'16
-- Collection of functions to build a 
local dl = require 'dataload'
local txt_load_util = {}
-- text utility functions
-- tokens contain all the characters from the whole sequence representing our document.
-- We are going to build a table containing all the different unique tokens
-- This will be our vocabulary. The place of each token (which is pseudo random) will generate the character id for the token.
-- This contains all the information about the building blocks,
-- The raw data can thus be represented as linear combinations of vectors with dimensions #vocab
function txt_load_util.buildVocab(tokens)
    assert(torch.type(tokens) == 'table', 'Expecting table')
    assert(torch.type(tokens[1]) == 'string', 'Expecting table of strings')
    local vocab,ivocab = {}
    for i=1,#tokens do
        local char = tokens[i]
		if vocab.char == nil then
			i = charid
			table.insert(ivocab,char)
			vocab.char = charid
		end 
	end
	return vocab, ivocab
end

function txt_load_util.text2tensor(ivocab)
    -- Build a tensor with a length which corresponds to size of vocabulary
    -- Each element in the vocab will receive an entry in this one-dimensional tensor
    local tensor = torch.IntTensor(#vocab):fill(0)
    -- Each number receives word id which corresponds to its frequency
    for i, char in ipairs(vocab) do
        tensor[i] = 1
    end

    return tensor
end

return txt_load_util
