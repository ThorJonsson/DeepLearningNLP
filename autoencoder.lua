-- Parameters
-- Date: 24/07/14

require 'rnn'

cmd = torch.CmdLine()
cmd:text()
cmd:text('Simple LSTM example for the RNN library')
cmd:text()
cmd:text('Options')
cmd:option('-use_saved',false,'Use previously saved inputs and trained network instead of new')
cmd:option('-cuda',true,'Run on CUDA-enabled GPU instead of CPU')
cmd:text()

-- parse input params
opt = cmd:parse(arg)

if opt.cuda then
   require 'cunn'
end         

-- Keep the input layer small so the model trains / converges quickly while training
local inputSize = 20
-- Larger numbers here mean more complex problems can be solved, but can also over-fit. 256 works well for now
local hiddenSize = 512
-- We want the network to classify the inputs using a one-hot representation of the outputs
--
local outputSize = 3

-- the dataset size is the total number of examples we want to present to the LSTM 
local dsSize=2000

-- We present the dataset to the network in batches where batchSize << dsSize
local batchSize=5

--And seqLength is the length of each sequence, i.e. the number of "events" we want to pass to the LSTM
--to make up a single example. I'd like this to be dynamic ideally for the YOOCHOOSE dataset..
local seqLength=50

-- number of target classes or labels, needs to be the same as outputSize above
-- or we get the dreaded "ClassNLLCriterion.lua:46: Assertion `cur_target >= 0 && cur_target < n_classes' failed. "
local nClass = 3
function build_network(inputSize, hiddenSize, outputSize)
   if opt.use_saved then
      rnn = torch.load('trained-model.t7')
   else
      rnn = nn.Sequential() 
      :add(nn.Linear(inputSize, hiddenSize))
      :add(nn.LSTM(hiddenSize, hiddenSize))
      :add(nn.LSTM(hiddenSize, hiddenSize))
      :add(nn.Linear(hiddenSize, outputSize))
      :add(nn.LogSoftMax())
      -- wrap this in a Sequencer such that we can forward/backward 
      -- entire sequences of length seqLength at once
      rnn = nn.Sequencer(rnn)
      if opt.cuda then
         rnn:cuda()
      end
   end
   return rnn
end

function build_input()
    torch.randn()
end

function build_data()
    local inputs = {}
    local targets = {}
    -- TODO in case we have an already trained model
    for i = 1, dsSize do
         -- populate both tables to get ready for training
         local input = torch.randn(batch_size,seq_len)
         local target = factors(input)
         if opt.cuda then
            input = input:float():cuda()
            target = target:float():cuda()
         end
         table.insert(inputs, input)
         table.insert(targets, target)
   end
   return inputs, targets
   end
-- Synthetic data:
-- We choose to exploit prime factorization with our neural network
-- This will show us how far the LSTM is from complicated deterministic tasks

-- **Prime Factorization**
-- For a given random number this function returns the Prime factors
-- The prime factors are stored in a table


function factor(num)
    local i = 2;
    local factors = {};

    if not num or num < 1 then
        print('your input must be postive!')
    end

    if num and num == 1 then
        factors[1] = 1
        return factors
    end

    while num and num > 1 do
        while num % i == 0 do
            factors[#factors + 1] = i
            num = num / i
        end
        i = i + 1
    end

    return factors
end



-- two tables to hold the *full* dataset input and target tensors
local inputs, targets = build_data()
local rnn = build_network(inputSize, hiddenSize, outputSize)


