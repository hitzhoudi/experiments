require 'gnuplot'

function split(line, sep)
    local t, i = {}, 1
    for word in string.gmatch(line, "([^" .. sep .. "]+)") do
        t[i] = tonumber(word); i = i + 1
    end
    return t
end

layer_1 = {}
layer_2 = {}
layer_3 = {}
layer_4 = {}
for line in io.lines("./sigmoid/dweight_file") do
    tokens = split(line, '\t')
    layer_1[#layer_1+1] = tokens[1]
    layer_2[#layer_2+1] = tokens[2]
    layer_3[#layer_3+1] = tokens[3]
    layer_4[#layer_4+1] = tokens[4]
end

gnuplot.epsfigure('sigmoid_vanishing_grad')
gnuplot.title('Vanishing Gradient of Sigmoid')
gnuplot.grid(true)
gnuplot.axis{0, '', 0, 0.006}
gnuplot.plot({'layer_1', torch.Tensor(layer_1), '~'}, {'layer_2', torch.Tensor(layer_2), '~'}, {'layer_3', torch.Tensor(layer_3), '~'}, {'layer_4', torch.Tensor(layer_4), '~'})
gnuplot.plotflush()
