
class config:
	project_dir = '/home/alange/workspace/dr/topics1/upstream'
	host_dir    = '/media/auto/alange-ms01-r/virtual/hostshare'
	vm_prefix   = 'test-load'
	vm_count    = 5

import sys
import collections

import numpy as np
import math
from scipy.stats import norm
from scipy.stats import logistic
import matplotlib.pyplot as plt

sys.path.append(config.project_dir)
import util
import guestmodel

def getVMFile(n):
	#return '{}/profile/test-load{}/stats-1'.format(config.host_dir, i)
	#return '{}/plots/profile/test-load{}/stats-1'.format(config.project_dir, i)
	return '{}/plots/profile-data5/test-load{}/stats-1'.format(config.project_dir, i)

class PotentialModel:
	_cpus         = None # physical CPUs
	_vcpus        = None # total VCPUs
	_guest_vcpus  = None

	def __init__(self, cpus, vcpus, guest_vcpus):
		assert cpus > 0
		assert vcpus >= guest_vcpus
		self._cpus         = cpus
		self._vcpus        = vcpus
		self._guest_vcpus  = guest_vcpus

	def minima(self):
		if self._cpus >= self._vcpus: return (1., 1., 0.)
		pr = float(self._vcpus % self._cpus) / self._cpus
		min1 = 1. / math.ceil(self._vcpus / self._cpus)
		min2 = 1. / math.floor(self._vcpus / self._cpus)
		return (pr, min1, min2)

	def potential_min_sf(self, x):
		pr, min1, min2 = self.minima()
		return pr * norm.sf(x, loc=min1, scale=.02) + \
			(1 - pr) * norm.sf(x, loc=min2, scale=.02)

	def potential_d_sf(self, other_demand, x):
		assert (other_demand + self._guest_vcpus) <= self._vcpus
		p = round(float(self._cpus) / float(other_demand + self._guest_vcpus), 3)
		if p > self._guest_vcpus:
			p = float(self._guest_vcpus)
		return norm.sf(x, loc=p, scale=.04)
	
	def potential_sf(self, other_demand, x):
		assert (other_demand + self._guest_vcpus) <= self._vcpus
		other_vcpus = self._vcpus - self._guest_vcpus
		pr = float(other_vcpus - other_demand) / other_vcpus \
			if other_vcpus > 0 and other_demand > (self._cpus - self._guest_vcpus) \
			else 1.
		m = self.potential_min_sf(x)
		d = self.potential_d_sf(other_demand, x)
		return (1. - pr) * m + pr * d

'''

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
	vm1_otherdemand.append(m_line['other_demand'] * 100. / m_line['cpus'])
	vm1_predict.append(
		100. * m.predictRatio(
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
#############################################################
# Tests


#def sigmoid(p,x):
#	x0,y0,c,k=p
#	y = c / (1 + np.exp(-k*(x-x0))) + y0
#	return y
#
#def minima_sig(minima, x):
#	pr, min1, min2 = minima
#	p_min1 = (min1, 0.0, 1, -50)
#	p_min2 = (min2, 0.0, 1, -50)
#	return pr * sigmoid(p_min1, x) + \
#		(1 - pr) * sigmoid(p_min2, x)

#x = np.linspace(0,1,100)
#plt.plot(x, sigmoid(p_min1,x), '-')
#plt.plot(x, sigmoid(p_min2,x), '-')
#plt.plot(x, minima_sig(minima(3,7), x), '-')


x = np.linspace(0,1,100)
y = PotentialModel(3,6,1)
demand = 3.2

# Plot the results
plt.plot(x, y.potential_min_sf(x),       '-', label='min')
plt.plot(x, y.potential_d_sf(demand, x), '-', label='d')
plt.plot(x, y.potential_sf(demand, x),   '-', label='p')

plt.xlabel('vcpu time')
plt.ylabel('probability')
plt.grid(True)
plt.show()

#'''
