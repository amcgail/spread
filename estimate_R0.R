require('EpiEstim')
library(readr)

# for debugging benefit
args = list('tmp.csv')
args = unlist(args)

{
  args  <- commandArgs(trailingOnly = TRUE)
  
  infection_timings <- readLines(args[1])
  infection_timings <- as.numeric(unlist(infection_timings))
  
  #print(infection_timings)
  
  res = estimate_R(infection_timings, method='parametric_si', config=make_config(list(mean_si=2.6, std_si=1.5)))
  writeLines(format_csv(res$R), stdout())
};