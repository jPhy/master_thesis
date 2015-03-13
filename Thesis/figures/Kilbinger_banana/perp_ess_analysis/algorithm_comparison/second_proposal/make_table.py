#! /usr/bin/python
import numpy as np

# load data
data_few_components  = np.loadtxt('original_algorithm_5_components.txt' , skiprows=1)
data_many_components = np.loadtxt('original_algorithm_15_components.txt', skiprows=1)
data_vb              = np.loadtxt('vb_only.txt'                         , skiprows=1)


# calculate mean values
mean_few_components  = np.mean(data_few_components , axis=0)
mean_many_components = np.mean(data_many_components, axis=0)
mean_vb              = np.mean(data_vb             , axis=0)

# calcualte variances
cov_few_components  = np.diag(np.cov(data_few_components , rowvar=0))
cov_many_components = np.diag(np.cov(data_many_components, rowvar=0))
cov_vb              = np.diag(np.cov(data_vb             , rowvar=0))

# calculate standard deviations
std_deviation_few_components  = np.sqrt(cov_few_components )
std_deviation_many_components = np.sqrt(cov_many_components)
std_deviation_vb              = np.sqrt(cov_vb             )

data_meaning = "chain_groups components_first perp_first ess_first components_final perp_final ess_final N_importance"


outfile = open('final_table.txt', 'w')

outfile.write("original_algorithm_5_components: ")
outfile.write("\n" + data_meaning)
outfile.write("\n means: " + str(mean_few_components))
outfile.write("\n std_dev: " + str(std_deviation_few_components))

outfile.write('\n\n\n\n\n')

outfile.write("original_algorithm_15_components: ")
outfile.write("\n" + data_meaning)
outfile.write("\n means: " + str(mean_many_components))
outfile.write("\n std_dev: " + str(std_deviation_many_components))

outfile.write('\n\n\n\n\n')

outfile.write("vb_only: ")
outfile.write("\n" + data_meaning)
outfile.write("\n means: " + str(mean_vb))
outfile.write("\n std_dev: " + str(std_deviation_vb))


outfile.close()
