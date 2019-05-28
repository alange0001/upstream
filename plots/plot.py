
import sys
import collections

from mpl_toolkits.mplot3d.axes3d import Axes3D
import numpy
import math
from scipy.stats import norm
from scipy.stats import logistic
import matplotlib.pyplot as plt


#'''

class config:
	project_dir = '/home/alange/workspace/dr/topics1/upstream'
	host_dir    = '/media/auto/alange-ms01-r/virtual/hostshare'
	vm_prefix   = 'test-load'
	vm_count    = 5
	plot_telemetry = True

sys.path.append(config.project_dir)
import util
import guestmodel

def getVMFile(n):
	#return '{}/profile/test-load{}/stats-1'.format(config.host_dir, i)
	return '{}/plots/profile-cpus2-vms5_6/test-load{}/stats-1'.format(config.project_dir, i)

#############################################################
# Usage, Steal, and Demand

if config.plot_telemetry:
	fig, axs = plt.subplots(config.vm_count, 1)
	fig.set_figheight(7)

for i in range(1,config.vm_count+1):
	name    = '{}{}'.format(config.vm_prefix, i)
	vm_file = getVMFile(i)

	log = util.WatchLog(vm_file, 'json')
	lines = log.read()

	for j in range(0,7):
		del lines[0]
		del lines[-1]

	stats = [ guestmodel.GuestStats(l) for l in lines ]

	time_min = stats[0].time
	x = [ s.time - time_min for s in stats ]
	usage  = [ s.guest_usage  for s in stats ]
	steal  = [ s.guest_steal  for s in stats ]
	demand = [ s.guest_demand for s in stats ]

	if i == 1:
		vm1_stats = stats
		vm1_x = x
		vm1_usage = usage

	if config.plot_telemetry:
		axs[i-1].set_ylim(bottom=-0.05, top=1.1)
		axs[i-1].plot(x, usage,  lw=1)
		axs[i-1].plot(x, steal,  lw=1)
		axs[i-1].plot(x, demand, lw=1)

if config.plot_telemetry:
	fig.legend(loc='lower center', labels=['usage', 'steal', 'VM\'s demand'], ncol=3, frameon=True)
	plt.show()

for j in range(len(vm1_stats)-1, -1, -1):
	if vm1_stats[j].guest_demand < .98:
		del vm1_stats[j]
		del vm1_x[j]
		del vm1_usage[j]


#############################################################
# Model
fig, ax = plt.subplots(1, 1)
fig.set_figheight(4)

#m = guestmodel.Model('', 300)
m2 = guestmodel.GuestModel(
		vm1_stats[0].host_cpus,
		vm1_stats[0].total_vcpus,
		vm1_stats[0].guest_vcpus)

vm1_otherdemand = []
vm1_m2p1        = []
vm1_m2L        = []
vm1_m2U        = []

for s in vm1_stats:
	other_demand = s.total_demand - s.guest_demand
	vm1_otherdemand.append(other_demand)
	vm1_m2p1.append(m2.predict1(other_demand))
	vm1_m2L.append(m2.predictLower(other_demand))
	vm1_m2U.append(m2.predictUpper(other_demand))

#ax.set_ylim(bottom=-5, top=110)
ax.plot(vm1_otherdemand, vm1_usage,   'o', lw=1)
ax.plot(vm1_otherdemand, vm1_m2p1, 'o', lw=1, color='green')
ax.plot(vm1_otherdemand, vm1_m2L, 'o', lw=1, color='orange')
ax.plot(vm1_otherdemand, vm1_m2U, 'o', lw=1, color='orange')

vm1_fit_x = []
vm1_fit_y = []
for i in range(0,len(vm1_otherdemand)):
	if vm1_otherdemand[i] > (vm1_stats[0].host_cpus * .9):
		vm1_fit_x.append(vm1_otherdemand[i])
		vm1_fit_y.append(vm1_usage[i])
a, b, c = numpy.polyfit(vm1_fit_x, vm1_fit_y, 2)
print(a,b,c)
vm1_otherdemand = numpy.array(vm1_otherdemand)
ax.plot(vm1_otherdemand, c + b * vm1_otherdemand + a * vm1_otherdemand**2, 'o', lw=1, color='gray')


fig.legend(loc='lower center', labels=['usage', 'modelR', 'modelL', 'modelU'], ncol=4, frameon=True)
plt.show()


#############################################################
fig, ax = plt.subplots(1, 1)
fig.set_figheight(4)

ax.plot(vm1_otherdemand, numpy.array([0 for i in range(0,len(vm1_otherdemand))]), '-', lw=1, color='black',)
ax.plot(vm1_otherdemand, numpy.array(vm1_usage) - numpy.array(vm1_m2L), 'o', lw=1, color='red',)

#hist, xedges, yedges = numpy.histogram2d(vm1_otherdemand, vm1_usage, bins=(20,20))
#count_up = [ 0 for i in range(0,len(hist)) ]
#count_down = [ 0 for i in range(0,len(hist)) ]
#for i in range(0,len(hist)):
#	model = m2.predict1(	xedges[i])
#	#print(model)
#	for j in range(-1,len(yedges)):
#		if j < (len(yedges)-2) and yedges[j+1] > model:
#			break
#	#print(i, j)
#	sum_hist_i = float(sum(hist[i]))
#	if sum_hist_i > 0:
#		if j >= 0:
#				count_down[i] = sum(hist[i][:j]) / sum_hist_i
#				count_up[i]   = sum(hist[i][j:]) / sum_hist_i
#		else:
#			count_down[i] = 0.
#			count_up[i]   = 1.

#axs[1].plot(xedges[:-1], count_up, lw=1)
#axs[1].plot(xedges[:-1], count_down, lw=1)

fig.legend(loc='lower center', labels=['reference', 'error'], ncol=2, frameon=True)
plt.show()

fig, ax = plt.subplots(1, 1)
fig.set_figheight(4)
ax.plot(vm1_otherdemand, numpy.array([0 for i in range(0,len(vm1_otherdemand))]), '-', lw=1, color='black',)
ax.plot(vm1_otherdemand, numpy.array(vm1_usage) - numpy.array(vm1_m2U), 'o', lw=1, color='red',)
fig.legend(loc='lower center', labels=['reference', 'error'], ncol=2, frameon=True)
plt.show()


#############################################################
#fig = plt.figure()
#ax = fig.add_subplot(111, projection='3d')
#
#hist, xedges, yedges = numpy.histogram2d(vm1_otherdemand, vm1_usage, bins=(20,20))
#for i in hist:
#	s = float(sum(i))
#	for j in range(0,len(i)):
#		i[j] = i[j] / s
#xpos, ypos = numpy.meshgrid(xedges[:-1], yedges[:-1], indexing="ij")
#xpos = xpos.ravel()
#ypos = ypos.ravel()
#zpos = 0
#dx = xedges[1] - xedges[0]
#dy = yedges[1] - yedges[0]
#dz = hist.ravel()
#
#ax.bar3d( xpos, ypos, zpos , dx, dy, dz, zsort='average')
#
#plt.show()


'''
#############################################################
# Tests


#def sigmoid(p,x):
#	x0,y0,c,k=p
#	y = c / (1 + numpy.exp(-k*(x-x0))) + y0
#	return y
#
#def minima_sig(minima, x):
#	pr, min1, min2 = minima
#	p_min1 = (min1, 0.0, 1, -50)
#	p_min2 = (min2, 0.0, 1, -50)
#	return pr * sigmoid(p_min1, x) + \
#		(1 - pr) * sigmoid(p_min2, x)

#x = numpy.linspace(0,1,100)
#plt.plot(x, sigmoid(p_min1,x), '-')
#plt.plot(x, sigmoid(p_min2,x), '-')
#plt.plot(x, minima_sig(minima(3,7), x), '-')


fig, ax = plt.subplots(1, 1)

x = numpy.linspace(0.1,4,50)
y = 1/x + x

# Plot the results
plt.plot(x, y, '-', label='min')

#t = y.potential_ppf(demand, x)

#'''
