# https://towardsdatascience.com/animating-your-data-visualizations-like-a-boss-using-r-f94ae20843e3

hand_data <- read.csv(file="log_1616588440.log",head=TRUE,sep=" ")
hand_data <- transform(hand_data, timestamp = (timestamp - hand_data[1,1]))
hand_data <- transform(hand_data, height = (height * 0.001))
hand_data

library(plotly)

p <- hand_data %>%
  plot_ly(
    x = ~x, 
    y = ~y, 
    size = 10, 
    color = ~height, 
    frame = ~timestamp, 
    text = ~height, 
    hoverinfo = "text",
    type = 'scatter',
    mode = 'markers'
  ) %>%
  layout(
    xaxis = list(
      type = "log"
    )
  )