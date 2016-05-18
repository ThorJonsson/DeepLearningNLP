-- Thor H. Jonsson
-- Trains the network
-- Based on examples for RNN package
local train = {}

function train.run(xplog)
    -- The current epoch of this model
    local epoch = xplog.epoch
    -- trainset object from dl package
    local trainset = xplog.trainset
    -- same for validset
    local validset = xplog.validset
    -- The module that gives us the target we are looking for
    local targetmodule = xplog.targetmodule
    -- The criterion we minimize 
    local criterion = xplog.criterion
    -- The model we are using
    local model = xplog.model
    --[[ 
    The following are the hyperparameters, we keep them in a table called opt 
    
    startlr', 0.05, 'learning rate at t=0')
    minlr', 0.00001, 'minimum learning rate')
    saturate', 400, 'epoch at which linear decayed LR will reach minlr')
    schedule', '', 'learning rate schedule. e.g. {[5] = 0.004, [6] = 0.001}')
    momentum', 0.9, 'momentum')
    maxnormout', -1, 'max l2-norm of each layer\'s output neuron weights')
    cutoff', -1, 'max l2-norm of concatenation of all gradParam tensors')
    cuda', false, 'use CUDA')
    device', 1, 'sets the device (GPU) to use')
    maxepoch', 1000, 'maximum number of epochs to run')
    earlystop', 50, 'maximum number of epochs to wait to find a better local minima for early-stopping')
    progress',true, 'print progress bar')
    silent', false, 'don\'t print anything to stdout')
    uniform', 0.1, 'initialize parameters using uniform distribution between -uniform and uniform. -1 means default initialization')
    
    lstm', true, 'use Long Short Term Memory (nn.LSTM instead of nn.Recurrent)')
    gru', false, 'use Gated Recurrent Units (nn.GRU instead of nn.Recurrent)')
    seqlen', 40, 'sequence length : back-propagate through time (BPTT) for this many time-steps')
    hiddensize', '{200}', 'number of hidden units used at output of each recurrent layer. When more than one is specified, RNN/LSTMs/GRUs are stacked')
    dropout', 0, 'apply dropout with this probability after each rnn layer. dropout <= 0 disables it.')
    
    batchsize', 32, 'number of examples per batch')
    trainsize', 4000, 'number of train examples seen between each epoch')
    validsize', 2400, 'number of valid examples used for early stopping and cross-validation') 
    savepath', './charResults/', 'path to directory where experiment log (includes model) will be saved')
    id', 'althingi_blstm', 'id string of this experiment (used to name output file) (defaults to a unique id)')
    ]]
    local opt = xplog.opt
    local ntrial = 0
    paths.mkdir(opt.savepath)
    opt.lr = opt.startlr

    -- Here we train and validate
    while opt.maxepoch <= 0 or epoch <= opt.maxepoch do
        print("")
        print("Epoch #"..epoch.." :")
    
       -- 1. training
       -- set timer to follow progress in training
       local a = torch.Timer()
       -- Set the model to training()
       -- the network remembers all previous rho (number of time-steps) states. This is necessary for BPTT.
       model:training()
       local sumErr = 0
        
       -- subiter takes in two arguments, i.e. batchsize and epochsize
       -- inputs : seqlen x batchsize [x inputsize] TODO I get batchsize x seqlen
       -- targets : seqlen x batchsize [x inputsize] 64 x 32 x 15 
       for i, inputs, targets in trainset:subiter(opt.seqlen, opt.trainsize) do --TODO opt.seqlen instead for batchsize????
           -- forward the target through criterion
           -- Double cast is needed because of convolution
       
           targets = targetmodule:forward(targets) -- intTensor with dimension 32 x 32 
           -- forward the model to obtain an output batch
           local outputs = model:forward(inputs)
           -- Given an input and a target, compute the loss function associated to the criterion and return the result. 
           -- In general input and target are Tensors, but some specific criterions might require some other type of object.
           -- This is a batch of training data, each row in the tensor representing the batch will be used for training
           local err = criterion:forward(outputs, targets)
           sumErr = sumErr + err

           -- backpropagate, here we correcting the weights to account for errors
           -- Given an input and a target, compute the gradients of the loss function associated to the criterion and return
           -- the result. input, target and gradInput are Tensors
           local gradOutputs = criterion:backward(outputs, targets)
           model:zeroGradParameters()
           model:backward(inputs, gradOutputs)

           -- TODO We could maybe use cutoff here?
           model:updateGradParameters(opt.momentum) -- affects gradParams
           model:updateParameters(opt.lr) -- affects params
           model:maxParamNorm(opt.maxnormout) -- affects params

           -- I always want to see the progress
           -- Later we should store info like this in a json file
           xlua.progress(math.min(i + opt.seqlen, opt.trainsize), opt.trainsize)

           if i % 1000 == 0 then
               collectgarbage()
           end

       end

       -- learning rate decay
       if opt.schedule then
           opt.lr = opt.schedule[epoch] or opt.lr
       else
           opt.lr = opt.lr + (opt.minlr - opt.startlr)/opt.saturate
       end
       opt.lr = math.max(opt.minlr, opt.lr)

       print("learning rate", opt.lr)
       -- TODO look at meannorm here?

       -- CUDA ONLY
       if cutorch then cutorch.synchronize() end

       -- Gives the speed for each batch
       local speed = a:time().real/opt.trainsize
       print(string.format("Speed : %f sec/batch ", speed))

       -- Gives the perplexity TODO for words
       local ppl = torch.exp(sumErr/opt.trainsize)
       print("Training PPL : "..ppl)

       -- save the perplexity in a table of perplexities for each epoch
       xplog.trainppl[epoch] = ppl

       -- 2. cross-validation

       -- We evaluate the model
       -- Everything is pretty much the same as in the training. 
       model:evaluate()
       local sumErr = 0
       for i, inputs, targets in validset:subiter(opt.seqlen, opt.validsize) do
           targets = targetmodule:forward(targets)
           local outputs = model:forward(inputs)
           local err = criterion:forward(outputs, targets)
           sumErr = sumErr + err
       end

       local ppl = torch.exp(sumErr/opt.validsize)
       print("Validation PPL : "..ppl)

       xplog.valppl[epoch] = ppl
       ntrial = ntrial + 1

       -- early-stopping
       if ppl < xplog.minvalppl then
           -- save best version of model
           xplog.minvalppl = ppl
           xplog.epoch = epoch 
           xplog.model = model
           local filename = paths.concat(opt.savepath, opt.id..'.t7')
           print("Found new minima. Saving to "..filename)
           torch.save(filename, xplog)
           ntrial = 0
       elseif ntrial >= opt.earlystop then
           print("No new minima found after "..ntrial.." epochs.")
           print("Stopping experiment.")
           break
       end

       collectgarbage()
       epoch = epoch + 1
   end
   print("Evaluate model using : ")
   print("th scripts/evaluate-rnnlm.lua --xplogpath "..paths.concat(opt.savepath, opt.id..'.t7')..(opt.cuda and '--cuda' or ''))
   return xplog
end
return train
