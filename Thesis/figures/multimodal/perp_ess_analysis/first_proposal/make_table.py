#! /usr/bin/python
import numpy as np

# load data
data_very_few_samples  = np.loadtxt('perp_ess_first_proposals_1250samples_before_mc_adapt.txt', skiprows=1)
data_a_million_samples = np.loadtxt('perp_ess_first_proposals_5000samples_before_mc_adapt.txt', skiprows=1)
data_very_many_samples = np.loadtxt('perp_ess_first_proposals_7500samples_before_mc_adapt.txt', skiprows=1)


# calculate mean values
mean_very_few_samples  = np.mean(data_very_few_samples , axis=0)
mean_a_million_samples = np.mean(data_a_million_samples, axis=0)
mean_very_many_samples = np.mean(data_very_many_samples, axis=0)

# calcualte variances
cov_very_few_samples  = np.diag(np.cov(data_very_few_samples , rowvar=0))
cov_a_million_samples = np.diag(np.cov(data_a_million_samples, rowvar=0))
cov_very_many_samples = np.diag(np.cov(data_very_many_samples, rowvar=0))

# calculate standard deviations
std_deviation_very_few_samples  = np.sqrt(cov_very_few_samples )
std_deviation_a_million_samples = np.sqrt(cov_a_million_samples)
std_deviation_very_many_samples = np.sqrt(cov_very_many_samples)

data_meaning = "chain_groups components_PMC Perplexity_PMC ESS_PMC components_VB Perplexity_VB ESS_VB components_HC Perplexity_HC ESS_HC"


outfile = open('final_table.txt', 'w')

outfile.write("1250 samples: ")
outfile.write("\n" + data_meaning)
outfile.write("\n means: " + str(mean_very_few_samples))
outfile.write("\n std_dev: " + str(std_deviation_very_few_samples))

outfile.write('\n\n\n\n\n')
outfile.write("5000 samples: ")
outfile.write("\n" + data_meaning)
outfile.write("\n means: " + str(mean_a_million_samples))
outfile.write("\n std_dev: " + str(std_deviation_a_million_samples))

outfile.write('\n\n\n\n\n')
outfile.write("7500 samples: ")
outfile.write("\n" + data_meaning)
outfile.write("\n means: " + str(mean_very_many_samples))
outfile.write("\n std_dev: " + str(std_deviation_very_many_samples))


outfile.close()
