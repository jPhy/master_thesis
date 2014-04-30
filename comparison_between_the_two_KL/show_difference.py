"""This file illustrates the difference between KL(q||P) and KL(P||q)
It uses a Student T function for P and tries to minimize the two KL
with a Gaussian q.

"""

from sys import stdout
import pypmc
import numpy as np
from scipy.stats import loggamma
from math import exp
from matplotlib import pyplot as plt

student_t_target = pypmc.density.student_t.StudentT(np.zeros(1), np.array([[.1]]), 2.1)
loggamma_target  = lambda x: loggamma.pdf(-x, (.1), loc=50.)

# in order to use PMC and GaussianInference, need to contruct a Gaussian
# mixture with one component
initial_guess = pypmc.density.mixture.create_gaussian_mixture(np.array([[-40.]]), np.array([[[1.]]]))

stdout.write('generating student t data ...'); stdout.flush()
data_student_t = student_t_target.propose(10**5)
stdout.write('done\n')

stdout.write('generating loggamma data ...'); stdout.flush()
mc_loggamma = pypmc.sampler.markov_chain.MarkovChain(lambda x: np.log(loggamma_target(x)), start=np.array([-40.]),
                                                     proposal=pypmc.density.gauss.LocalGauss(np.eye(1)*10.))
mc_loggamma.run(10**5)
data_loggamma = mc_loggamma.history[:]
print 'done'

# ------------------------------ KL(q||P) ------------------------------
# this KL is minimized by variational Bayes

stdout.write('running VB ...'); stdout.flush()

vb_student_t = pypmc.mix_adapt.variational.GaussianInference(data_student_t, initial_guess=initial_guess)
vb_student_t.run()
q_P_student_t = vb_student_t.make_mixture()

vb_loggamma = pypmc.mix_adapt.variational.GaussianInference(data_loggamma, initial_guess=initial_guess)
vb_loggamma.run()
q_P_loggamma = vb_loggamma.make_mixture()

print 'done'

# ------------------------------ KL(P||q) ------------------------------
# this KL is minimized by PMC

stdout.write('running PMC ...'); stdout.flush()

P_q_student_t = P_q_loggamma = initial_guess

for i in range(20):
    P_q_student_t = pypmc.mix_adapt.pmc.gaussian_pmc(data_student_t, P_q_student_t)
    P_q_loggamma  = pypmc.mix_adapt.pmc.gaussian_pmc(data_loggamma , P_q_loggamma)

print 'done'

# ------------------------------ plot results ------------------------------
stdout.write('plotting ...'); stdout.flush()

# Student T
plotpoints = np.linspace(-5, 5, 10**5).reshape(10**5,1)
plt.figure()
plt.title('Student T')
plt.plot(plotpoints, map(lambda x: exp(student_t_target.evaluate(x)), plotpoints), label='target')
plt.plot(plotpoints, map(lambda x: exp(P_q_student_t.evaluate(x)), plotpoints), label='P_q PMC')
plt.plot(plotpoints, map(lambda x: exp(q_P_student_t.evaluate(x)), plotpoints), label='q_P VB')
plt.legend()

# loggamma
plotpoints = np.linspace(-60, 20, 10**5).reshape(10**5,1)
plt.figure()
plt.title('log gamma')
plt.plot(plotpoints, map(lambda x: loggamma_target(x), plotpoints), label='target')
plt.plot(plotpoints, map(lambda x: exp(P_q_loggamma.evaluate(x)), plotpoints), label='P_q PMC')
plt.plot(plotpoints, map(lambda x: exp(q_P_loggamma.evaluate(x)), plotpoints), label='q_P VB')
plt.legend()

print 'done'

plt.show()
