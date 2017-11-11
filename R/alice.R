library(lubridate)
library(readr)

df <- read_csv('/Users/andrew/reps/mlcourse_open/data/alice/train/Alice_log.csv', 
               col_names = TRUE,
               col_types = list(
                 timestamp = col_datetime(format = ""),
                 site = col_character()
               ))

df$hours <- hour(df$timestamp)
df$mins <- hour(df$timestamp) * 60 + minute(df$timestamp)

ggplot(df) + geom_col(aes(mins, 1))
# how to draw X scale with hours and min

ggplot(df) + geom_col(aes(hours, 1)) + scale_x_continuous(breaks = seq(0, 24, 1), 
                                                          labels = as.character(seq(0, 24, 1)))

min(df$timestamp)
max(df$timestamp)

