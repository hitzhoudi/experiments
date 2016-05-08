data <- read.csv(file="./relu/hidden_output_file",sep="\t",head=FALSE)
layer1 <- data[,1]
layer2 <- data[,2]
layer3 <- data[,3]
layer4 <- data[,4]
layer1_density <- density(layer1)
plot(layer1_density)
hist(layer1)
layer2_density <- density(layer2)
plot(layer2_density)
hist(layer2)
layer3_density <- density(layer3)
plot(layer3_density)
hist(layer3)
layer4_density <- density(layer4)
plot(layer4_density)
hist(layer4)
