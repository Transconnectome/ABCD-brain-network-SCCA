### load packages
library(ggseg)
#library(ggseg3d)
library(ggplot2)
library(repr)

# Arguments for parallel computing
args <- commandArgs(TRUE)
domain <- as.character(args[1])
exp_tag <- as.character(args[2])

cor <- read.csv('/storage/connectome/seojw/data/DTI_node_coordinate_22_07_14.csv')[, c(2, 3, 9)]

bootstrap_data_dir = sprintf('/storage/connectome/seojw/data/PMD_boot_result/%s_BNM/first_comp_cov/%s/summary_result', domain, exp_tag)

BNM_metric_full_list <- list('norm_clust_coef', 'norm_Cc', 'deg', 'stren','BC'
			   #  'zstat_kcore', 'zstat_score', 'within_module_deg', 
			   #  'clust_coef', 'kcore', 'score', 'participation_coef', 'Cc'
			     )

for (CCA_metric in list('cross_loading', 'weight', 'loading', 'density_partial_loading', 'density_partial_cross_loading')){
	for (comp_num in 1:5){
    data <- read.csv(sprintf('%s/bootstrap_result_summary_y_%s_comp%d.csv', bootstrap_data_dir, CCA_metric, comp_num))
    
    if (domain == 'GPS' & comp_num == 2){
    	data[, 2:10] = -1 * data[,2:10]
    }
    var_num = length(data[, 1])    # count variable numbers for frequently occured variable selection
    occurence_rate = data[, 11]
    p = sum(occurence_rate) / var_num
    variable_selection_crit <- qbinom(1 - 0.001/var_num, 5000, p) / 5000
    print(variable_selection_crit)
    i = 1
    BNM_metric_list <- list()
    BNM_metric_name_list <- list()
    for (BNM_metric in BNM_metric_full_list){
      if (nrow(data[grep(BNM_metric, data$X), ]) != 0){
        BNM_metric_list[i] <- list(BNM_metric=data[grep(BNM_metric, data$X), ])
        BNM_metric_name_list[i] <- BNM_metric
        data <- data[-grep(BNM_metric, data$X), ]
        i = i + 1
      }
    }
    
    for (metric_num in 1:length(BNM_metric_list)){
      metric <- BNM_metric_list[metric_num][[1]]
      
      # change index name format from {BNM_metric}_{ROI} to {ROI}
      for (i in 1:length(metric[, 1])){
        metric_split <- strsplit(metric$X[i], '_')[[1]]
        metric$X[i] <- metric_split[length(metric_split)]
      }
      
      colnames(metric)[1] = 'feat'
      metric$feat <- sub('-','.',metric$feat)
      aligned_metric <- merge(cor, metric, by='feat')
  
      
      sig_68 <- aligned_metric[, 5] * aligned_metric[, 6] > 0
      sig_95 <- aligned_metric[, 7] * aligned_metric[, 8] > 0
      sig_99 <- aligned_metric[, 9] * aligned_metric[, 10] > 0
      sig_997 <- aligned_metric[, 11] * aligned_metric[, 12] > 0
      consistency <- aligned_metric[, 13]
      consistent_cond <- consistency >= variable_selection_crit
      
      cond_list = list(sig_68 & consistent_cond, sig_95 & consistent_cond, sig_99 & consistent_cond, sig_997 & consistent_cond)
      cond_name_list = list('68', '95', '99', '99.7')
      
      cortex = aligned_metric[, 3] == 1
      subcort = aligned_metric[, 3] == 0
      
      # visualize significant BNM metric on the brain image  (cortical region)
      for (sig_lev_num in 2:2){
        cort_cond <- cond_list[sig_lev_num][[1]] & cortex
        subcort_cond <- cond_list[sig_lev_num][[1]] & subcort
        
        cort_label = aligned_metric[cort_cond, 2]
        if (CCA_metric == 'loading'){
          color_uplim = 1
          color_downlim = -1
        } else if(CCA_metric == 'cross_loading') {
          color_uplim = 0.1
          color_downlim = -0.1
	} else {
	  color_uplim = 0.25
	  color_downlm = -0.25
        }
        

        # create output directory
        output_dir = sprintf('%s/BNM_result/comp%d/ci_%s/%s', bootstrap_data_dir, comp_num, cond_name_list[sig_lev_num], CCA_metric)
        if (!dir.exists(output_dir)){
          dir.create(output_dir, recursive=TRUE)
        }
                      
        if (length(cort_label) != 0){
          cort_loading = aligned_metric[cort_cond, 4]
          metric_data <- data.frame(label=cort_label, loading=cort_loading, stringAsFactors = FALSE)
          (ggseg(.data=metric_data, mapping=aes(fill=loading), color = 'white', position = "stacked", adapt_scales = TRUE) 
                + ggtitle(sprintf('%s, %s, %s, %s, comp%d, CI = %s %%', domain, exp_tag, BNM_metric_name_list[[metric_num]], CCA_metric, comp_num, cond_name_list[sig_lev_num]))
                +scale_fill_gradient2(limits=c(color_downlim, color_uplim), low='blue', high='red', mid='gray', midpoint=0) + theme(title = element_text(size = 10)))
          options(repr.plot.width=15, repr.plot.height=10)
          saving_file_name = sprintf('%s/%s_consistBN', output_dir, BNM_metric_name_list[[metric_num]])
          ggsave(sprintf('%s.png', saving_file_name))
          ggsave(sprintf('%s.pdf', saving_file_name))
          
        }
        
        
        # plot subcortical regions
        subcort_label = aligned_metric[subcort_cond, 2]
        if (length(subcort_label) != 0){
          subcort_loading = aligned_metric[subcort_cond, 4]
          metric_data <- data.frame(label=subcort_label, loading=subcort_loading, atlas='aseg', stringAsFactors = FALSE)
          (ggseg(.data=metric_data, atlas='aseg', mapping=aes(fill=loading), color = 'white', position = "stacked", adapt_scales = TRUE) 
                + ggtitle(sprintf('%s, %s, %s, %s, comp%d, CI = %s %%', domain, exp_tag, BNM_metric_name_list[[metric_num]], CCA_metric, comp_num, cond_name_list[sig_lev_num]))
                +scale_fill_gradient2(limits=c(color_downlim, color_uplim), low='blue', high='red', mid='gray', midpoint=0) + theme(title = element_text(size = 8)))
          options(repr.plot.width=15, repr.plot.height=10)
          
          saving_file_name = sprintf('%s/%s_consistBN_subcort', output_dir, BNM_metric_name_list[[metric_num]])
          ggsave(sprintf('%s.png', saving_file_name))
          ggsave(sprintf('%s.pdf', saving_file_name))
        }
      }
    }
  }
}


