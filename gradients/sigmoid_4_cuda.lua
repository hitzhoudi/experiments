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
    local record = torch.DoubleTensor(300*1000, #model.modules/2-1):cuda()
    local correct, incorrect = 0, 0
    for i=1, #data do
        local y = model:forward(data[i][1]:cuda())
        _, idx = torch.max(y, 1)
        if idx[1] == data[i][2][1] then
            correct = correct + 1
        else
            incorrect = incorrect + 1
        end
        for m=2, #model.modules-1, 2 do
            record[{{(i-1)*1000+1, i*1000}, m/2}] = model.modules[m].output
        end
    end
    return correct, incorrect, {torch.mean(record, 1), torch.std(record, 1)}
end

-- model
model = nn.Sequential()
model:add(nn.Linear(28*28, 1000))
model:add(nn.Sigmoid())
model:add(nn.Linear(1000, 1000))
model:add(nn.Sigmoid())
model:add(nn.Linear(1000, 1000))
model:add(nn.Sigmoid())
model:add(nn.Linear(1000, 1000))
model:add(nn.Sigmoid())
model:add(nn.Linear(1000, 10))
model:add(nn.LogSoftMax())
criterion = nn.ClassNLLCriterion()
model:cuda()
weights, dweights = model:parameters()
criterion:cuda()

--read data
train_data = read_data("../data/mnist_train.csv")
valid_data = read_data("../data/mnist_valid_small.csv")
print("Data is loaded")

os.execute("mkdir sigmoid")
local ls_file = io.open('sigmoid/loss_file', 'w')
local ac_file = io.open('sigmoid/accuracy_file', 'w')
local hd_file = io.open('sigmoid/hidden_output_file', 'w')

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
        model:zeroGradParameters()
    end
    if cur % 20000 == 19999 then
        print(cur .. " samples are done")
        --print loss information
        ls_file:write(string.format("loss in last 20k cases: %f\n", batch_loss/20000))
        batch_loss = 0
        --validation
        local correct, incorrect, record = evaluate(model, valid_data)
        ac_file:write(string.format("correct rate in validation: %f\n", correct/(correct+incorrect)))
        hd_file:write(string.format("%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\n",
            record[1][1][1], record[1][1][2], record[1][1][3], record[1][1][4],
            record[2][1][1], record[2][1][2], record[2][1][3], record[2][1][4]))
    end
    cur = cur + 1
end

ls_file:close()
ac_file:close()

