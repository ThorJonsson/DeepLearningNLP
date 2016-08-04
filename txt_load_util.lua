-- Thor Jonsson 26/05'16
-- Collection of functions to build a 
local dl = require 'dataload'
local txt_load_util = {}
-- text utility functions

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

function txt_load_util.tokenize(text)
    local keywords = {'<eos>','<msw>'}
    local tokens = {}
    local tmp = {}
    local utf8 = require 'lua-utf8'
    -- First we do the characters
    i = 0 
    local keyword_candidate = false
    local keyword_found = false
    for c in utf8.gmatch(text,'.') do -- splits into characters
        -- Do nothing, this character is evil. It is reserved for dummy moses.
        if c == '_' then
        elseif c == '<' and keyword_candidate then 
            keyword_candidate = false
            for c in tmp do
                table.insert(tokens,c)
            end
            tmp = {}
        elseif c == '<' then
            table.insert(tmp,c)
            keyword_candidate = true
        elseif c == '>' and keyword_candidate then
            table.insert(tmp,c)
            tmp_txt = table.concat(tmp)
            for _,keyword in pairs(keywords) do
                if tmp_txt == keyword then
                    table.insert(tokens,tmp_txt)
                    keyword_found = true
                    tmp = {}
                end
            end
            if not keyword_found then
                for _,c in pairs(tmp) do
                    table.insert(tokens,c)
                end
                tmp = {}
            end
            keyword_found = false
            keyword_candidate = false
        elseif keyword_candidate then
            table.insert(tmp,c)
        else
            table.insert(tokens,c)
        end
    end
    return tokens
end

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



return txt_load_util
