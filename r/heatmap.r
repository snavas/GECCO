hand_data <- read.csv(file="D:/paula/Documents/NotFun/Studium/Master_Geoinformatics/GECCO/GECCO/logs/log.log",head=TRUE,sep=" ")
hand_data <- transform(hand_data, timestamp = (timestamp - hand_data[1,1]))
hand_data <- transform(hand_data, height = (height * 0.001))

################################################################################
library(ggplot2)
ggplot(hand_data, aes(x = x, y = 720-y)) +
  coord_equal() + 
  xlab('x') + 
  ylab('y') + 
  xlim(0, 1080) +
  ylim(0, 720) +
  stat_density2d(aes(fill = ..level..), alpha = .5,
                 geom = "polygon", data = hand_data) + 
  scale_fill_viridis_c() + 
  theme(legend.position = 'right') +
  geom_point(aes(x,720-y), size=5,shape=".")

ggplot(hand_data, aes(x = x, y = 720-y)) +
  coord_equal() + 
  xlab('x') + 
  ylab('y') + 
  xlim(0, 1080) +
  ylim(0, 720) +
  stat_density2d(aes(color = ..level..), alpha = .5,
                 geom = "polygon", data = hand_data) + 
  scale_fill_viridis_c() + 
  theme(legend.position = 'right') +
  geom_point(aes(x,720-y), size=5,shape=".")

ggplot(hand_data, aes(x = x, y = 720-y)) +
  geom_point(aes(x,720-y), size=5,shape=".") +
  coord_equal() + 
  xlab('x') + 
  ylab('y') + 
  xlim(0, 1080) +
  ylim(0, 720) +
  stat_density2d(aes(fill= ..density..), alpha = .5,
                 geom = "raster", data = hand_data, contour=FALSE) + 
  scale_fill_viridis_c() + 
  theme(legend.position = 'right')
  
ggplot(hand_data, aes(x = x, y = 720-y)) +
  coord_equal() + 
  xlab('x') + 
  ylab('y') + 
  xlim(0, 1080) +
  ylim(0, 720) +
  stat_density2d(aes(fill= ..density..), alpha = .5,
                 geom = "raster", data = hand_data, contour=FALSE, h=c(300,300)) + 
  scale_fill_viridis_c() + 
  theme(legend.position = 'right')
