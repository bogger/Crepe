--[[ Tester for Crepe
By Xiang Zhang @ New York University
--]]

require("sys")

local Extract = torch.class("ExtractGrad")

-- Initialization of the testing script
-- data: Testing dataset
-- model: Testing model
-- config: (optional) the configuration table
function ExtractGrad:__init(data,model)
   

   -- Store the objects
   self.data = data
   self.model = model
   self.max_label = 2
   --self.loss = loss

   -- Move the type
   --self.loss:type(model:type())

   -- Create time table
   self.time = {}


end

-- Execute testing for a batch step
function ExtractGrad:run()
   -- Initializing the errors and losses
   
   self.output = {}
   self.output[1]= torch.FloatTensor()
   self.output[2] = torch.FloatTensor()
   -- Start the loop
   self.clock = sys.clock()
   self.params, self.grads = self.model:getParameters()
   count=0
   for batch,idx,labels,n in self.data:iterator() do
      count=count+1
      print('minibatch number:'..count)
      self.batch = self.batch or batch:transpose(2,3):contiguous():type(self.model:type())
      self.labels = self.labels or labels:type(self.model:type())
      self.idx = self.idx or idx:type(self.model:type())
      self.batch:copy(batch:transpose(2, 3):contiguous())
      self.idx:copy(idx)
      self.labels:copy(labels)
      --max_label = torch.max(labels)
      sp_len = self.idx:size(2)
      -- Record time
      if self.model:type() == "torch.CudaTensor" then cutorch.synchronize() end
      self.time.data = sys.clock() - self.clock

      self.clock = sys.clock()
      -- Forward propagation
      --print(self.output:size())
      --print(self.model:forward(self.batch):size())
      self.model:forward(self.batch)
      --print(self.batch:size())
      --print(torch.type(self.grads))
      self.grads:zero()
      local gradOutput = torch.zeros(n, self.max_label):type('torch.CudaTensor')
      local i
      --print(labels)
      --print(n)
      for i=1,n do
         gradOutput[i][self.labels[i]] = 1
      end
      local grads = self.model:backward(self.batch, gradOutput)
      --print(grads:size())
      local grads_new = torch.zeros(n,sp_len)
      for i = 1,n do
            for j=1,sp_len do
               if self.idx[i][j] > 0 then
                  grads_new[i][j] = grads[i][j][self.idx[i][j]]
               end
            end
      end
      --gradOutput = torch.zeros(max_label,n):type('torch.CudaTensor') --testing
      if self.output[1]:numel()==0 then
         
         self.output[1] = grads_new
         
         self.output[2]= self.labels
      else
         
         self.output[1] = torch.cat(self.output[1],grads_new,1)
         self.output[2] = torch.cat(self.output[2],self.labels)
      end
      -- Record time
      if self.model:type() == "torch.CudaTensor" then cutorch.synchronize() end
      self.time.forward = sys.clock() - self.clock

      self.clock = sys.clock()      
     
      
   end
   return self.output
end
