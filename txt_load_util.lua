-- Thor Jonsson 26/05'16
-- Collection of functions to build a 
local txt_load_util = {}
-- text utility functions

function txt_load_util.buildVocab(tokens, ischartokens, minfreq)
    assert(torch.type(tokens) == 'table', 'Expecting table')
    assert(torch.type(tokens[1]) == 'string', 'Expecting table of strings')
    minfreq = minfreq or -1
    assert(torch.type(minfreq) == 'number')
    local wordfreq = {}

    -- Build a table containing each token and the respective frequency in data
    for i=1,#tokens do
        local word = tokens[i]
        wordfreq[word] = (wordfreq[word] or 0) + 1
    end

    local vocab, ivocab = {}, {}
    local wordseq = 0
    
    local _ = require 'moses'
    -- make sure ordering is consistent
    local words = _.sort(_.keys(wordfreq))
    -- Get UTF8 support to determine length of words
    local utf8 = require 'lua-utf8'
    local maxwordlen = 0
    local oov = 0
    for i, word in ipairs(words) do
        local freq = wordfreq[word]
        if freq >= minfreq then
            if not ischartokens and utf8.len(word) > maxwordlen then
                maxwordlen = utf8.len(word)
            end
            wordseq = wordseq + 1
            vocab[word] = wordseq
            ivocab[wordseq] = word
        else
            oov = oov + freq
        end
    end

    if oov > 0 then
        wordseq = wordfreq + 1
        wordfreq['<OOV>'] = oov
        vocab['<OOV>'] = wordseq
        ivocab[wordseq] = '<OOV>'
    end
    if ischartokens then
        return vocab, ivocab, wordfreq
    else
        return vocab, ivocab, wordfreq, maxwordlen
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

return text_load_util
