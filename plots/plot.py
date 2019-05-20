
project_dir = '/home/alange/workspace/dr/topics1/upstream'

import sys
sys.path.append(project_dir)
import util
import collections

import numpy
from scipy.stats import logistic
import matplotlib.pyplot as plt

'''
####### examples ########
fig, ax = plt.subplots(1, 1)

mean, var, skew, kurt = logistic.stats(moments='mvsk')

x = numpy.linspace(logistic.ppf(0.01),
		logistic.ppf(0.99), 100)

ax.plot(x, logistic.pdf(x),
		'r-', lw=5, alpha=0.6, label='logistic pdf')

rv = logistic()
ax.plot(x, rv.pdf(x), 'k-', lw=2, label='frozen pdf')


vals = logistic.ppf([0.001, 0.5, 0.999])
numpy.allclose([0.001, 0.5, 0.999], logistic.cdf(vals))

r = logistic.rvs(size=1000)

ax.hist(r, density=True, histtype='stepfilled', alpha=0.2)
ax.legend(loc='best', frameon=False)
#'''

'''
####### tests ########
loc=5
scale=1
x = numpy.linspace(0, 10, 50)
ax.plot(x, logistic.sf(x, loc=loc, scale=scale), 'k-', lw=2, label='cdf')
#'''


vm_count = 5

fig, axs = plt.subplots(vm_count, 1)
fig.set_figheight(7)

for i in range(1,vm_count+1):
	name    = 'test-load{}'.format(i)
	#vm_file = '/media/auto/alange-ms01-r/virtual/hostshare/profile/test-load{}/stats-1'.format(i)
	vm_file = '/media/auto/alange-ms01-r/virtual/hostshare/usr_global/src/upstream/plots/profile-data2/test-load{}/stats-1'.format(i)

	log = util.WatchLog(vm_file, 'json')
	lines = log.read()

	time_min = int(lines[0]['time'])
	x = [ int(l['time']) - time_min for l in lines ]
	usage = [ float(l['vm']['vcpu'][0]['usage']) for l in lines ]
	steal = [ float(l['vm']['vcpu'][0]['steal']) for l in lines ]
	demand = [ float(l['vm']['vcpu'][0]['usage']) + float(l['vm']['vcpu'][0]['steal']) for l in lines ]

	axs[i-1].set_ylim(ymin=-5, ymax=110)
	axs[i-1].plot(x, usage,  lw=1)
	axs[i-1].plot(x, steal,  lw=1)
	axs[i-1].plot(x, demand, lw=1)

fig.legend(loc='lower center', labels=['usage', 'steal', 'VM\'s demand'], ncol=3, frameon=True)
plt.show()
