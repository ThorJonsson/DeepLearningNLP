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

-- We add this function to txt_load_util
function string:split(inSplitPattern, outResults)
    if not outResults then
        outResults = {}
    end
    local theStart = 1
    local theSplitStart, theSplitEnd = string.find(self, inSplitPattern, theStart)
    while theSplitStart do
        table.insert(outResults, string.sub(self, theStart, theSplitStart-1))
        theStart = theSplitEnd +1
        theSplitStart, theSplitEnd = string.find(self, inSplitPattern, theStart)
    end
    return outResults
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


