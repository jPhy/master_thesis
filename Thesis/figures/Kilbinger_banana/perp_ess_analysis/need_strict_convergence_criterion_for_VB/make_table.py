#! /usr/bin/python
import numpy as np

# load data
data_strict = np.loadtxt('calculate_perp_ess_VB_strict_convergence.txt', skiprows=1)
data_loose  = np.loadtxt('calculate_perp_ess_VB_loose_convergence.txt', skiprows=1)


# calculate mean values
mean_strict = np.mean(data_strict, axis=0)
mean_loose  = np.mean(data_loose , axis=0)

# calcualte variances
cov_strict = np.diag(np.cov(data_strict, rowvar=0))
cov_loose  = np.diag(np.cov(data_loose , rowvar=0))

# calculate standard deviations
std_deviation_strict = np.sqrt(cov_strict)
std_deviation_loose  = np.sqrt(cov_loose )

data_meaning = "chain_groups components_VB Perplexity_VB ESS_VB"


outfile = open('final_table.txt', 'w')

outfile.write("strict convergence criterion: ")
outfile.write("\n" + data_meaning)
outfile.write("\n means: " + str(mean_strict))
outfile.write("\n std_dev: " + str(std_deviation_strict))

outfile.write('\n\n\n\n\n')

outfile.write("loose convergence criterion: ")
outfile.write("\n" + data_meaning)
outfile.write("\n means: " + str(mean_loose))
outfile.write("\n std_dev: " + str(std_deviation_loose))


outfile.close()
