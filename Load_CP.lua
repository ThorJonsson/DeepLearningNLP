require 'torch' --Torch tensor classes
require 'nn'  --Neural Network Modules
require 'nngraph'
require 'optim'
require 'lfs'
require 'util.Squeeze'
require 'util.misc'
require 'util.HLogSoftMax'
local dbg = require('debugger.lua/debugger')

HSMClass = require 'util.HSMClass'
BatchLoader = require 'util.BatchLoaderUnk'
model_utils = require 'util.model_utils'

local stringx = require('pl.stringx')
HighwayMLP = require 'model.HighwayMLP'
TDNN = require 'model.TDNN'
LSTMTDNN = require 'model.LSTMTDNN'

-- Requires the packages necessary for using the GPU
function ignite_gpu()
    io.write('Now igniting the GPU')
    -- check that cunn/cutorch are installed if user wants to use the GPU
    local ok, cunn = pcall(require, 'cunn')
    local ok2, cutorch = pcall(require, 'cutorch')
    if not ok then print('package cunn not found!') end
    if not ok2 then print('package cutorch not found!') end
    if ok and ok2 then
        print('using CUDA on GPU ')
        print('Make sure that your saved checkpoint was also trained with GPU. If it was trained with CPU use -gpuid -1 for sampling as well')
        cutorch.setDevice(1) -- note +1 to make it 0 indexed! sigh lua
        cutorch.manualSeed(123)
    else
        print('Falling back on CPU mode')
        --opt.gpuid = -1 -- overwrite user setting
    end
end

-- Loads the checkpoint
function load()
  ignite_gpu()
  local checkpoint
  local answer
  answer ='lm_word-small_epoch25.00_1484.83.t7'
  io.write('Choosing to load checkpoint: '..answer..' \n')
  checkpoint = torch.load(answer)
  return checkpoint
end


-- This function returns a table which contains two empty cuda tensors 
-- for each layer. I don't know why lstm networks need two of these tensors for each layer
function init_curr_state(checkpoint)
  local current_state = {}
  for layer=1, checkpoint.opt.num_layers do
    local h_init = torch.zeros(1,checkpoint.opt.rnn_size):double():cuda()
    table.insert(current_state,h_init:clone())
    table.insert(current_state,h_init:clone())
  end
  return current_state
end

-- Utility function to split a string into words
function split_string(input_str)
  local table = {}
  local i = 1
  -- Explanation on the pattern comes later I found it on stackexchange
  for word in string.gmatch(input_str,"([^%s]+)") do 
    table[i] = word
    i = i+1
  end
  return table
end

-- Asks user to give input to seed
function ask4input()
  io.write('Give the computer an input to correct: \n')
  io.flush()
  local answer = io.read()
  -- split the answer into words and put it in a table
  return split_string(answer)
end
-- Row-normalize a matrix
function normalize(m)
  m_norm = torch.zeros(m:size())
  for i = 1, m:size(1) do
    m_norm[i] = m[i] / torch.norm(m[i])
  end
  return m_norm
end

function get_word_embeddings(checkpoint)
  return checkpoint.protos.rnn.modules[2].weight:double()
end

-- Return the k-nearest words to a word or a vector based on cosine similarity
-- w can be a string such as "king" or a vector for ("king" - "queen" + "man")
function get_sim_words(w, k, checkpoint, word2idx, idx2word)
    word_vecs = get_word_embeddings(checkpoint)
    if word_vecs_norm == nil then
        word_vecs_norm = normalize(word_vecs)
    end
    if type(w) == "string" then
        if word2idx[w] == nil then
           print("'"..w.."' does not exist in vocabulary.")
           return nil
        else
            w = word_vecs_norm[word2idx[w]]
        end
    end
    -- Compute the matrix vector product
    local sim = torch.mv(word_vecs_norm, w)
    sim, idx = torch.sort(-sim)
    local r = {}
    for i = 1, k do
        r[i] = {idx2word[idx[i]], -sim[i]}
    end
    return r
end



---print similar words
function print_sim_words(words, k, checkpoint, word2idx, idx2word)
    for i = 1, #words do
        r = get_sim_words(words[i], k, checkpoint, word2idx, idx2word)
        if r ~= nil then
            print("-------"..words[i].."-------")
            for j = 1, k do
                print(string.format("%s, %.4f", r[j][1], r[j][2]))
            end
        end
    end
end


function forward_update_states(prev_word, gmod)
    --dbg()
    local lst = gmod.rnn:forward{prev_word, unpack(current_state)}
    -- lst is a list of [state1,state2,..,stateN,output]. 
    -- The output is the log probabilities, i.e. element #lst
    -- Now lst contains current state of the nn
    local current_state = {} -- reset
    for i=1, state_size do table.insert(current_state, lst[i]) end -- clone
    return lst[#lst]
end


-- The following function should only be called from inside sample_nn
-- This is because the vocab and the module variables are global 
-- If we change this we need to pass these variables to this function
-- Before: current state initialized, vocab mappings global.
function seed_nn(gmod)
  -- We ask the user for input and then tokenize that input on words and put it in a table 
  local seed_table = ask4input()
  local prev_word
  -- Before: vocab mapping unpacked, nn module with weights is gmod
  -- After: We have made a forward pass based on the word prev_word and recorded it in current_state
  for j=1,#seed_table do
    local word = seed_table[j]
    -- TODO: if word does not exist in vocab what shall we do?
    prev_word = torch.Tensor{word2idx[word]}
    -- Make a forward pass based on the word given 
    print(prev_word)
  end
  local prediction = forward_update_states(prev_word, gmod)
  local prediction = lst[#lst]
  return prediction 
end


function predicted_words(prediction)
    local answer
    prediction:div(1) 
    probs = torch.exp(prediction):squeeze()
    probs:div(torch.sum(probs))
    local words = torch.multinomial(probs:float(),5):resize(5):float()
    io.write('Choose a number corresponding to the word you prefer to be next? \n')
    for i=1,5 do io.write(i..': '..idx2word[words[i]]..'\n') end
    io.flush()
    repeat 
      answer = io.read()
    until 0 < tonumber(answer) and tonumber(answer) < 5 
    io.write(idx2word[words[answer]])
    return words[answer]
end

-- Seeds the network
function sample_gamaltnn()
  ignite_gpu()
  local checkpoint = torch.load('lm_ptb-char-small_epoch10.00_111.20.t7')
  -- This are the different vocabularies, inverted and not both for character and word level
  local idx2word, word2idx, idx2char, char2idx = table.unpack(checkpoint.vocab)
  
  local protos = checkpoint.protos
  protos.rnn:evaluate() -- To make use of dropout
  
  local current_state = {}
  for layer=1, checkpoint.opt.num_layers do
    local h_init = torch.zeros(2,checkpoint.opt.rnn_size):cuda()
    table.insert(current_state,h_init:clone())
    table.insert(current_state,h_init:clone())
  end
  local prev_char = torch.Tensor{word2idx['yes']}:cuda()
  local lst = protos.rnn:forward{prev_char, unpack(current_state)}
end
  -- Before: Current_state is zeroTemporalMaxPooling.lua:15: bad argument #2 to 'TemporalMaxPooling_updateOutput' (input sequence smaller than kernel size)
  -- After: Seed_input has been generated, the nn has been forwarded based on those inputs
  -- From that we get a probabilities for different words in the vocabulary 
  --local prediction = seed_nn(gmod)
  --for i=1,200 do
  --  if i%10 == 0 then io.write('\n') end
  --  prev_word = predicted_words(prediction)
  --  prediction = forward_update_states(prev_word,gmod)
  --end
  --io.write('\n')
  --io.flush()
  
--end
-- Seeds the network
function sample_nn(checkpoint)
  -- For convenience: The variables for the vocab are global
  -- This are the different vocabularies, inverted and not both for character and word level
  idx2word, word2idx, idx2char, char2idx = table.unpack(checkpoint.vocab)
  -- This contains is the nn module which we have trained
  -- It contains the weights 
  local gmod = checkpoint.protos
  gmod.rnn:evaluate() -- To make use of dropout
  -- Let's try to make it global
  current_state = init_curr_state(checkpoint)
  
  -- This should be kept global  
  state_size = #current_state
  -- Before: Current_state is zero
  -- After: Seed_input has been generated, the nn has been forwarded based on those inputs
  -- From that we get a probabilities for different words in the vocabulary 
  local prediction = seed_nn(gmod)
  for i=1,200 do
    if i%10 == 0 then io.write('\n') end
    prev_word = predicted_words(prediction)
    prediction = forward_update_states(prev_word,gmod)
  end
  io.write('\n')
  io.flush()
  
end
