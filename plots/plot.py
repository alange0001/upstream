
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
	plot_telemetry = False
	experiments    = [
			(2,4),
			(2,5),
			(2,6),
			(3,4),
			(3,5),
			(3,6),
			(3,7),
			(3,8),
			(3,9),
			]
	@classmethod
	def getVMFile(cls, cpus, vms, n):
		#return '{}/profile/test-load{}/stats-1'.format(config.host_dir, n)
		return '{}/plots/data3-intel/cpu{}-vm{}/test-load{}/stats-1'.format(cls.project_dir, cpus, vms, n)

sys.path.append(config.project_dir)
import util
import guestmodel

class VMStats:
	host_cpus    = None
	total_vcpus  = None
	guest_vcpus  = None
	stats        = None
	time         = None

	def __init__(self, cpus, vms, n):
		vm_file = config.getVMFile(cpus, vms, n)
		log = util.WatchLog(vm_file, 'json')
		lines = log.read()

		for j in range(0,0):
			del lines[0]
			del lines[-1]

		self.stats = stats = [ guestmodel.GuestStats(l) for l in lines ]

		time_min = stats[0].time
		self.time   = [ s.time - time_min for s in stats ]

		self.host_cpus   = stats[0].host_cpus
		self.total_vcpus = stats[0].total_vcpus
		self.guest_vcpus = stats[0].guest_vcpus

	def cutThreshold(self, value):
		for j in range(len(self.stats)-1, -1, -1):
			if self.stats[j].guest_demand < value:
				del self.stats[j]
				del self.time[j]
				self._cache_guest_usage  = None
				self._cache_guest_steal  = None
				self._cache_guest_demand = None
				self._cache_other_demand = None

	_cache_guest_usage = None
	@property
	def guest_usage(self):
		if self._cache_guest_usage is None:
			self._cache_guest_usage = numpy.array([ s.guest_usage  for s in self.stats ])
		return self._cache_guest_usage
	
	_cache_guest_steal = None
	@property
	def guest_steal(self):
		if self._cache_guest_steal is None:
			self._cache_guest_steal = numpy.array([ s.guest_steal  for s in self.stats ])
		return self._cache_guest_steal
	
	_cache_guest_demand = None
	@property
	def guest_demand(self):
		if self._cache_guest_demand is None:
			self._cache_guest_demand = numpy.array([ s.guest_demand  for s in self.stats ])
		return self._cache_guest_demand
	
	_cache_other_demand = None
	@property
	def other_demand(self):
		if self._cache_other_demand is None:
			self._cache_other_demand = numpy.array([ s.total_demand - s.guest_demand for s in self.stats ])
		return self._cache_other_demand
	
	_cache_other_demand_idx = None
	_cache_guest_usage_by_other_demand = None
	def _construct_other_demand_cache(self):
			work = dict()
			other_demand = self.other_demand
			usage = self.guest_usage
			for i in range(0,len(other_demand)):
				round_od = round(other_demand[i], 1)
				if work.get(round_od) is None: work[round_od] = []
				work[round_od].append(usage[i])
				
			self._cache_other_demand_idx = list(work.keys())
			self._cache_other_demand_idx.sort()
			self._cache_other_demand_idx = numpy.array(self._cache_other_demand_idx)
			self._cache_guest_usage_by_other_demand = [ numpy.array(work[i]) for i in self._cache_other_demand_idx ]
	@property
	def other_demand_idx(self):
		if self._cache_other_demand_idx is None:
			self._construct_other_demand_cache()
		return self._cache_other_demand_idx
	@property
	def guest_usage_by_other_demand(self):
		if self._cache_guest_usage_by_other_demand is None:
			self._construct_other_demand_cache()
		return self._cache_guest_usage_by_other_demand

	_cache_model = None
	@property
	def model(self):
		if self._cache_model == None:
			self._cache_model = guestmodel.GuestModel(
				self.host_cpus,
				self.total_vcpus,
				self.guest_vcpus)
		return self._cache_model


###########################################################################
###########################################################################
experiments = collections.OrderedDict()
for experiment_idx in config.experiments:

	experiment_data = VMStats(*experiment_idx, 1)
	experiments[experiment_idx] = experiment_data

	#############################################################
	# Usage, Steal, and Demand
	if config.plot_telemetry:
		fig, axs = plt.subplots(experiment_data.total_vcpus, 1)
		fig.set_figheight(7)

		for i in range(1, experiment_data.total_vcpus +1):
			stats = experiment_data if i == 1 else VMStats(*experiment_idx, i)
			axs[i-1].set_ylim(bottom=-0.05, top=1.1)
			axs[i-1].plot(stats.time, stats.guest_usage,  lw=1)
			axs[i-1].plot(stats.time, stats.guest_steal,  lw=1)
			axs[i-1].plot(stats.time, stats.guest_demand,  lw=1)

		fig.legend(loc='lower center', labels=['usage', 'steal', 'VM\'s demand'], ncol=3, frameon=True)
		plt.show()


	experiment_data.cutThreshold(.98)


#############################################################
# Data and Models per experiment
def printDataModels():
	for experiment_idx, experiment_data in experiments.items():
		fig, ax = plt.subplots(1, 1)
		fig.set_figheight(4)
		ax.set_title('Models: CPUs {}, VMs {}'.format(experiment_data.host_cpus, experiment_data.total_vcpus))


		guest_model = experiment_data.model
		#ax.set_ylim(bottom=-5, top=110)
		ax.plot(experiment_data.other_demand, experiment_data.guest_usage,   'o', lw=1, color="#447F95")
		ax.plot(experiment_data.other_demand_idx, [ guest_model.predict1(o) for o in experiment_data.other_demand_idx ], '-', lw=2.5, color='#07BA4F')
		ax.plot(experiment_data.other_demand_idx, [ guest_model.predictLower(o) for o in experiment_data.other_demand_idx ], '-', lw=2.5, color='#14C0CC')
		ax.plot(experiment_data.other_demand_idx, [ guest_model.predictUpper(o) for o in experiment_data.other_demand_idx ], '-', lw=2.5, color='#14C0CC')

		if False:
			vm1_fit_x = []
			vm1_fit_y = []
			for i in range(0,len(vm1_otherdemand)):
				if vm1_otherdemand[i] > (vm1_stats[0].host_cpus * .8):
					vm1_fit_x.append(vm1_otherdemand[i])
					vm1_fit_y.append(vm1_usage[i])
			a, b, c = numpy.polyfit(vm1_fit_x, vm1_fit_y, 2)
			#print(a,b,c)
			vm1_otherdemand = numpy.array(vm1_otherdemand)
			ax.plot(vm1_otherdemand, c + b * vm1_otherdemand + a * vm1_otherdemand**2, 'o', lw=1, color='gray')


		if True:
			ax.plot(experiment_data.other_demand_idx, numpy.array([ numpy.percentile(i, 50) for i in experiment_data.guest_usage_by_other_demand ]), '-', lw=2.5, color='orange')
			ax.plot(experiment_data.other_demand_idx, numpy.array([ numpy.percentile(i,  5) for i in experiment_data.guest_usage_by_other_demand ]), '-', lw=2.5, color='#EEBF67')
			ax.plot(experiment_data.other_demand_idx, numpy.array([ numpy.percentile(i, 95) for i in experiment_data.guest_usage_by_other_demand ]), '-', lw=2.5, color='#EEBF67')

		fig.legend(loc='lower center', labels=['usage', 'ref', 'lower', 'upper', 'p50', 'p05', 'p95'], ncol=4, frameon=True)
		plt.show()


		#############################################################
		# Error
		if True:
			fig, ax = plt.subplots(1, 1)
			fig.set_figheight(3)
			ax.set_title('Lower Model Error: CPUs {}, VMs {}'.format(experiment_data.host_cpus, experiment_data.total_vcpus))
			ax.plot(experiment_data.other_demand_idx, [0 for i in range(0,len(experiment_data.other_demand_idx))], '-', lw=1, color='black')
			ax.plot(experiment_data.other_demand, [ s.guest_usage - guest_model.predictLower(s.total_demand - s.guest_demand) for s in experiment_data.stats ], 'o', lw=1, color='red')
			fig.legend(loc='lower center', labels=['reference', 'error'], ncol=2, frameon=True)
			plt.show()
	
			fig, ax = plt.subplots(1, 1)
			fig.set_figheight(4)
			ax.set_title('Upper Model Error: CPUs {}, VMs {}'.format(experiment_data.host_cpus, experiment_data.total_vcpus))
			ax.plot(experiment_data.other_demand_idx, [0 for i in range(0,len(experiment_data.other_demand_idx))], '-', lw=1, color='black')
			ax.plot(experiment_data.other_demand, [ s.guest_usage - guest_model.predictUpper(s.total_demand - s.guest_demand) for s in experiment_data.stats ], 'o', lw=1, color='red')
			fig.legend(loc='lower center', labels=['reference', 'error'], ncol=2, frameon=True)
			plt.show()

	#############################################################
	# 3D histogram
	if False:
		fig = plt.figure()
		ax = fig.add_subplot(111, projection='3d')

		hist, xedges, yedges = numpy.histogram2d(vm1_otherdemand, vm1_usage, bins=(20,20))
		for i in hist:
			s = float(sum(i))
			for j in range(0,len(i)):
				i[j] = i[j] / s
		xpos, ypos = numpy.meshgrid(xedges[:-1], yedges[:-1], indexing="ij")
		xpos = xpos.ravel()
		ypos = ypos.ravel()
		zpos = 0
		dx = xedges[1] - xedges[0]
		dy = yedges[1] - yedges[0]
		dz = hist.ravel()

		ax.bar3d( xpos, ypos, zpos , dx, dy, dz, zsort='average')

		plt.show()

printDataModels()

#############################################################
# Summary of Experiments
def printSummary():
	fig, ax = plt.subplots(1, 1)
	fig.set_figheight(5)
	ax.set_ylabel('Ti')
	ax.set_xlabel('D\'i / N\'i')
	
	for experiment_idx, experiment_data in experiments.items():
		cpus, total_vcpus = experiment_idx
		guest_vcpus = experiment_data.guest_vcpus
		
		ax.plot(
				experiment_data.other_demand_idx/(total_vcpus -guest_vcpus),
				numpy.array([ numpy.percentile(i,  5) for i in experiment_data.guest_usage_by_other_demand ]),
				'-', lw=1, label='M={}, N={}'.format(cpus, total_vcpus))
		
	ax.legend(loc='best', frameon=False)
	plt.show()

#printSummary()

def sigmoid(p,x):
	x0,y0,c,k=p
	y = c / (1 + numpy.exp(-k*(x-x0))) + y0
	return y

def fitBegin():
	fig, ax = plt.subplots(1, 1)
	fig.set_figheight(5)
	ax.set_ylabel('min demand / V\'i')
	ax.set_xlabel('N / M')

	data_p05_ini = []
	data_p50_ini = []
	data_p95_ini = []
	for experiment_idx, experiment_data in experiments.items():
		cpus, total_vcpus = experiment_idx
		guest_vcpus = experiment_data.guest_vcpus
		
		idx = experiment_data.other_demand_idx/(total_vcpus -guest_vcpus)
		value_p05 = numpy.array([ numpy.percentile(i,  5) for i in experiment_data.guest_usage_by_other_demand ])
		for i in range(0, len(idx)):
			if value_p05[i] < .95:
				i2 = i-1 if i > 0 else 0
				data_p05_ini.append((float(total_vcpus)/cpus, idx[i2]))
				break
			
		value_p50 = numpy.array([ numpy.percentile(i, 50) for i in experiment_data.guest_usage_by_other_demand ])
		for i in range(0, len(idx)):
			if value_p50[i] < .95:
				i2 = i-1 if i > 0 else 0
				data_p50_ini.append((float(total_vcpus)/cpus, idx[i2]))
				break

		value_p95 = numpy.array([ numpy.percentile(i, 95) for i in experiment_data.guest_usage_by_other_demand ])
		for i in range(0, len(idx)):
			if value_p95[i] < .95:
				i2 = i-1 if i > 0 else 0
				data_p95_ini.append((float(total_vcpus)/cpus, idx[i2]))
				break

	def f(x, a, b):
		return [ 1./(a*i + b) if (a*i + b) != 0 else numpy.inf for i in x ]
	def fit(x, f, pa, pb):
		fit_sum = None
		fit_vals = None
		for a in pa:
			for b in pb:
				y2 = f(x, a, b)
				s = sum((y - y2)**2)
				if fit_sum is None or s < fit_sum:
					fit_sum = s
					fit_vals = (a, b)
		return fit_vals

	x2 = numpy.linspace(1.3, 4, 50)
	
	data_p05_ini.sort(key=lambda x: x[0])
	x = numpy.array([ d[0] for d in data_p05_ini ])
	y = numpy.array([ d[1] for d in data_p05_ini ])
	ax.plot( x, y,	'*', label='p05 ini', color='blue')

	a, b = fit(x, f, numpy.linspace(.2, 5, 20), numpy.linspace(-10, 10, 50))
	print('p05', a, b)
	y2 = f(x2, a, b)
	ax.plot(x2, [i if i <= 1 else 1 for i in y2], '-', label='p05 ini', color='darkblue')


	data_p50_ini.sort(key=lambda x: x[0])
	x = numpy.array([ d[0] for d in data_p50_ini ])
	y = numpy.array([ d[1] for d in data_p50_ini ])
	ax.plot( x, y,	'*', label='p50 ini', color='orange')

	a, b = fit(x, f, numpy.linspace(.2, 5, 20), numpy.linspace(-10, 10, 50))
	print('p50', a, b)
	y2 = f(x2, a, b)
	ax.plot(x2, [i if i <= 1 else 1 for i in y2], '-', label='p50 ini3', color='darkorange')


	data_p95_ini.sort(key=lambda x: x[0])
	x = numpy.array([ d[0] for d in data_p95_ini ])
	y = numpy.array([ d[1] for d in data_p95_ini ])
	ax.plot( x, y,	'*', label='p95 ini', color='red')

	a, b = fit(x, f, numpy.linspace(.2, 5, 20), numpy.linspace(-10, 10, 50))
	print('p95', a, b)
	y2 = f(x2, a, b)
	ax.plot(x2, [i if i <= 1 else 1 for i in y2], '-', label='p95 inif', color='darkred')

	ax.legend(loc='best', frameon=False)
	plt.show()

#fitBegin()


#'''
#############################################################
# Tests
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


#fig, ax = plt.subplots(1, 1)

#x = numpy.linspace(1.3,4,50)

# Plot the results
#ax.plot(x, 1./(9*numpy.log(x) -1.5), '-', label='1')
#ax.plot(x, 10./(2**x + 2*x), '-', label='2')
#ax.plot(x, 1./numpy.log(2*x), '-', label='3')

#ax.legend(loc='best', frameon=False)

#t = y.potential_ppf(demand, x)

#'''
