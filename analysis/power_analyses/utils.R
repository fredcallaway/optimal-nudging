# -- -- -- -- -- -- -- -- -- -- -- 
# -- -- -- -- -- -- -- -- -- -- -- 
#  Helper functions shared by multiple power analyses
# -- -- -- -- -- -- -- -- -- -- -- 
# -- -- -- -- -- -- -- -- -- -- --

# Generates a sample that matches the constraints in the experiment
balanced_sample = function(input_data,num_samples){
  unique_items = unique(input_data$balancer_id)
  if (num_samples %% length(unique_items)){
    stop('ERROR! NOT BALANCED!')
  }
  total_df = data.frame()
  inner_n = num_samples/length(unique_items)
  for (item_i in unique_items){
    curr_subset = subset(input_data,balancer_id==item_i)
    curr_df = sample_n(curr_subset,inner_n)
    total_df = rbind(total_df,curr_df)
  }
  return(total_df)
}

# Function that loops over sample_sizes and gets the power for each
power_analysis = function(sim_data,sample_sizes,num_sims){
  power_results = data.frame()
  for (sample_size_i in sample_sizes){
    sim_results = hypothesis_tests(sim_data,sample_size_i,num_sims) %>% 
      colMeans() %>% 
      as.data.frame.list()
    sim_results$sample_size = sample_size_i
    power_results=rbind(power_results,sim_results) 
  }
  return(power_results)
}