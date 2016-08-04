local sequence_loader = torch.class('sequence_loader')

function sequence_loader:__init(sequence, batchsize, bidirectional)
   assert(torch.isTensor(sequence))
   assert(torch.type(batchsize) == 'number')
   -- sequence is a tensor where the first dimension indexes time
   self.batchsize = batchsize
   self.bidirectional = bidirectional 
   local seqlen = sequence:size(1)
   local size = sequence:size():totable()
   table.remove(size, 1)
   assert(#size == sequence:dim() - 1)
   self.data = sequence.new()
   -- note that some data will be lost
   -- Number of batches
   local seqlen2 = torch.floor(seqlen / batchsize)
   -- seqlen2 x batchsize
   self.data = sequence:sub(1,seqlen2*batchsize):view(batchsize, seqlen2):t():contiguous()
end

return sequence_loader
