library("Hmisc")

data <- read.csv(file="./sigmoid/hidden_output_file",sep="\t",head=FALSE)
d0 = data.frame(
  x0 = c(1:nrow(data)),
  y0 = data[,1],
  sd0 = data[,5]
)
d1 = data.frame(
  x1 = c(1:nrow(data)),
  y1 = data[,2],
  sd1 = data[,6]
)
d2 = data.frame(
  x2 = c(1:nrow(data)),
  y2 = data[,3],
  sd2 = data[,7]
)
d3 = data.frame(
  x3 = c(1:nrow(data)),
  y3 = data[,4],
  sd3 = data[,8]
)

#add error bars (without adjusting yrange)
par(fg = "blue")
with(
  data = d0,
  expr = errbar(x0, y0, y0+sd0/2, y0-sd0/2, add=F, pch=1, cap=0, col='blue')
)
par(fg = "red")
with(
  data = d1,
  expr = errbar(x1, y1, y1+sd1/2, y1-sd1/2, add=F, pch=1, cap=0, col='red')
)
par(fg = "yellow")
with(
  data = d2,
  expr = errbar(x2, y2, y2+sd2/2, y2-sd2/2, add=F, pch=1, cap=0, col='yellow')
)
par(fg = "green")
with(
  data = d3,
  expr = errbar(x3, y3, y3+sd3/2, y3-sd3/2, add=F, pch=1, cap=0, col='green')
)

