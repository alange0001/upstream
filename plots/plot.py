
class config:
	project_dir = '/home/alange/workspace/dr/topics1/upstream'
	host_dir    = '/media/auto/alange-ms01-r/virtual/hostshare'
	vm_prefix   = 'test-load'
	vm_count    = 5

import sys
import collections

import numpy as np
from scipy.stats import logistic
import matplotlib.pyplot as plt

sys.path.append(config.project_dir)
import util
import guestmodel

def getVMFile(n):
	#return '{}/profile/test-load{}/stats-1'.format(config.host_dir, i)
	return '{}/plots/profile/test-load{}/stats-1'.format(config.project_dir, i)

#############################################################
# Usage, Steal, and Demand

fig, axs = plt.subplots(config.vm_count, 1)
fig.set_figheight(7)

for i in range(1,config.vm_count+1):
	name    = '{}{}'.format(config.vm_prefix, i)
	vm_file = getVMFile(i)

	log = util.WatchLog(vm_file, 'json')
	lines = log.read()

	time_min = int(lines[0]['time'])
	x = [ int(l['time']) - time_min for l in lines ]
	usage = [ float(l['vm']['vcpu'][0]['usage']) for l in lines ]
	steal = [ float(l['vm']['vcpu'][0]['steal']) for l in lines ]
	demand = [ float(l['vm']['vcpu'][0]['usage']) + float(l['vm']['vcpu'][0]['steal']) for l in lines ]

	if i == 1:
		vm1_lines = lines
		vm1_x = x
		vm1_usage = usage

	axs[i-1].set_ylim(bottom=-5, top=110)
	axs[i-1].plot(x, usage,  lw=1)
	axs[i-1].plot(x, steal,  lw=1)
	axs[i-1].plot(x, demand, lw=1)

fig.legend(loc='lower center', labels=['usage', 'steal', 'VM\'s demand'], ncol=3, frameon=True)
plt.show()

#############################################################
# Model
fig, ax = plt.subplots(1, 1)
fig.set_figheight(4)

m = guestmodel.Model('', 300)
vm1_otherdemand = []
vm1_predict     = []
for l in vm1_lines:
	m_line = m.feed(l)
	vm1_otherdemand.append(m_line['other_demand'] * 100.)
	vm1_predict.append(
		100. * m.predict(
			m_line['cpus'],
			l['vm']['vcpu_count'],
			m_line['other_demand']))

#ax.set_ylim(bottom=-5, top=110)
ax.plot(vm1_x, vm1_usage,       lw=1)
ax.plot(vm1_x, vm1_otherdemand, lw=1)
ax.plot(vm1_x, vm1_predict,     lw=1, color='red',)

fig.legend(loc='lower center', labels=['usage', 'others\' demand', 'model'], ncol=3, frameon=True)
plt.show()

#############################################################
fig, ax = plt.subplots(1, 1)
fig.set_figheight(4)

ax.plot(vm1_otherdemand, vm1_usage,   'o', lw=1)
ax.plot(vm1_otherdemand, vm1_predict, 'o', lw=1, color='red')

plt.show()


'''

loc=50
scale=5
fig, ax = plt.subplots(1, 1)
ax.set_ylim(bottom=-0.05, top=1.05)

x = np.linspace(0, 100, 100)

ax.plot(x, .6 + .4 * logistic.sf(x, loc=loc, scale=scale), lw=1)

plt.show()

#'''
