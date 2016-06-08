-- Thor Jonsson 26/05'16
-- Collection of functions to build a 
local dl = require 'dataload'
local txt_load_util = {}
-- text utility functions

function txt_load_util.buildVocab(tokens)
    local ischartokens = true -- For now we only work with characters
    assert(torch.type(tokens) == 'table', 'Expecting table')
    assert(torch.type(tokens[1]) == 'string', 'Expecting table of strings')
    minfreq = -1
    local charfreq = {}

    -- Build a table containing each token and the respective frequency in data
    -- Note that we have already tokenized at this stage, so each element
    -- of the tokens contains an icelandic utf8-coded character
    for i=1,#tokens do
        local char = tokens[i]
        charfreq[char] = (charfreq[char] or 0) + 1
    end

    local vocab, ivocab = {}, {}
    local charseq = 0
    
    local _ = require 'moses'
    -- make sure ordering is consistent
    local chars = _.sort(_.keys(charfreq))
    -- Get UTF8 support to determine length of words
    local oov = 0
    -- Count order of vocabulary
    for i, char in ipairs(chars) do
        local freq = charfreq[char]
            -- oov = order of vocabulary
            oov = oov + freq
    end

    -- If the vocabulary is not empty
    if oov > 0 then
        charseq = charfreq + 1
        charfreq['<OOV>'] = oov
        vocab['<OOV>'] = charseq
        ivocab[charseq] = '<OOV>'
    end
    if ischartokens then
        return vocab, ivocab, charfreq
    end
end

function txt_load_util.text2tensor(tokens, vocab)
    local oov = vocab['<OOV>']--TODO much better implementation exists
    -- Build a tensor with a length which corresponds to size of vocabulary
    -- Each element in the vocab will receive an entry in this one-dimensional tensor
    local tensor = torch.IntTensor(#tokens):fill(0)
    -- Each number receives word id which corresponds to its frequency
    for i, char in ipairs(tokens) do
        local charid = vocab[char] 
        if not charid then
            assert(oov)
            charid = oov
        end

        tensor[i] = charid
    end

    return tensor
end

return txt_load_util
