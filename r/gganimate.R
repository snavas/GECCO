#https://www.datanovia.com/en/blog/gganimate-how-to-create-plots-with-beautiful-animation-in-r/

hand_data <- read.csv(file="log_1616588440.log",head=TRUE,sep=" ")
hand_data <- transform(hand_data, timestamp = (timestamp - hand_data[1,1]))
hand_data <- transform(hand_data, height = (height * 0.001))
hand_data
#head(hand_data)

library(ggplot2)
library(gganimate)
#library(viridis)
theme_set(theme_bw())

p <- ggplot(
  hand_data, 
  aes(x = x, y = y, size = 10, colour = height)
  ) +
  geom_point(show.legend = FALSE, alpha = 0.7) +
  #scale_color_viridis_d() +
  #scale_size(range = c(2, 12)) +
  #scale_x_log10() +
  xlim(0, 1080) +
  ylim(0, 720) +
  labs(x = "width (1080p)", y = "height (720p)") +
  geom_point(aes(colour = height)) +
  scale_colour_gradientn(colours = rainbow(3))
  #scale_color_viridis(option = "C")
  #scale_x_continuous(breaks = seq(0, 1080, 100)) +
  #scale_y_continuous(breaks = seq(0, 720, 100))

p

animation <- p + transition_time(hand_data$timestamp, range = c(0L, max(hand_data$timestamp))) +
  labs(title = "t: {frame_time}") +
  shadow_wake(wake_length = 0.1, alpha = FALSE)
  #shadow_mark(alpha = 0.3, size = 0.5)


animate(animation, height = 720, width =1080) #, fps=3)
anim_save("Gapminder_example.gif")

#print(p)
#show(p)

hand_data
head(hand_data)

