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
    local txt = 'Ég fór út að labba.<eos>Hvernig var veðrið?<eos>Hvar er mamma?<eos>Hún var skelfingu lostin!<eos>'
-- splittar í setningar
    local sentences = txt:split('<eos>')
    return sentences
end
