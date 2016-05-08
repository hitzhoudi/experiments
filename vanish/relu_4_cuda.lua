require 'nn'
require 'cutorch'
require 'cunn'

function split(line, sep)
    local t, i = {}, 1
    for word in string.gmatch(line, "([^" .. sep .. "]+)") do
        t[i] = tonumber(word); i = i + 1
    end
    return t
end

function read_data(path)
    local data = {}
    for line in io.lines(path) do
        local tokens = split(line, ',')
        local y = torch.Tensor(1)
        y[1] = tokens[1] + 1
        local i = 1
        local x = torch.Tensor(28*28):apply(function()
            i = i + 1
            return tokens[i]
        end)
        data[#data+1] = {x, y}
    end
    return data
end

function evaluate(model, data)
    local correct, incorrect = 0, 0
    for i=1, #data do
        local y = model:forward(data[i][1]:cuda())
        _, idx = torch.max(y, 1)
        if idx[1] == data[i][2][1] then
            correct = correct + 1
        else
            incorrect = incorrect + 1
        end
    end
    return correct, incorrect 
end

-- model
model = nn.Sequential()
model:add(nn.Linear(28*28, 1000))
model:add(nn.ReLU())
model:add(nn.Linear(1000, 1000))
model:add(nn.ReLU())
model:add(nn.Linear(1000, 1000))
model:add(nn.ReLU())
model:add(nn.Linear(1000, 1000))
model:add(nn.ReLU())
model:add(nn.Linear(1000, 10))
model:add(nn.LogSoftMax())
criterion = nn.ClassNLLCriterion()
model:cuda()
weights, dweights = model:parameters()
criterion:cuda()

--read data
train_data = read_data("../data/mnist_train.csv")
valid_data = read_data("../data/mnist_valid.csv")
print("Data is loaded")

os.execute("mkdir relu")
local dw_file = io.open('relu/dweight_file', 'w')
local ls_file = io.open('relu/loss_file', 'w')
local ac_file = io.open('relu/accuracy_file', 'w')

--start to train
local min_batch = 10
local cur, batch_loss = 0, 0
model:zeroGradParameters()
while cur / 100000 <= 150 do
    local x, y = train_data[cur%#train_data+1][1]:cuda(), train_data[cur%#train_data+1][2]:cuda()
    local output = model:forward(x)
    batch_loss = batch_loss + criterion:forward(output, y)
    model:backward(x, criterion:backward(output, y))
    --update model
    if cur % min_batch == min_batch - 1 then
        for i=1, #weights do
            weights[i]:add(-1e-3, dweights[i])
        end
        dw1 = torch.cumsum(torch.cumsum(torch.abs(dweights[1])), 2)[1000][28*28]/1000/28/28/min_batch
        dw3 = torch.cumsum(torch.cumsum(torch.abs(dweights[3])), 2)[1000][1000]/1000/1000/min_batch
        dw5 = torch.cumsum(torch.cumsum(torch.abs(dweights[5])), 2)[1000][1000]/1000/1000/min_batch
        dw7 = torch.cumsum(torch.cumsum(torch.abs(dweights[7])), 2)[1000][1000]/1000/1000/min_batch
        dw_file:write(string.format("%f\t%f\t%f\t%f\n", dw1, dw3, dw5, dw7))
        model:zeroGradParameters()
    end
    if cur % 20000 == 19999 then
        print(cur .. " samples are done")
        --print loss information
        ls_file:write(string.format("loss in last 20k cases: %f\n", batch_loss/20000))
        batch_loss = 0
        --validation
        local correct, incorrect = evaluate(model, valid_data)
        ac_file:write(string.format("correct rate in validation: %f\n", correct/(correct+incorrect)))
    end
    cur = cur + 1
end

dw_file:close()
ls_file:close()
ac_file:close()

