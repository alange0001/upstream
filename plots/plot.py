
import sys
import collections

from mpl_toolkits.mplot3d.axes3d import Axes3D
import matplotlib.pyplot as plt

import numpy
import math
from scipy.optimize import curve_fit
from sklearn import linear_model

class config:
	project_dir = '/home/alange/workspace/dr/topics1/upstream'
	host_dir    = '/media/auto/alange-ms01-r/virtual/hostshare'
	vm_prefix   = 'test-load'
	experiments    = [
			(2,4),
			(2,5),
			(2,6),
			(2,7),
			(2,8),
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
		return '{}/plots/data4-intel/cpu{}-vm{}/test-load{}/stats-1'.format(cls.project_dir, cpus, vms, n)

sys.path.append(config.project_dir)
import util
import guestmodel

#'''

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
				self._cache_fitSigmoid = dict()

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
			other_demand = self.other_demand
			usage = self.guest_usage

			other_vcpus = self.total_vcpus - self.guest_vcpus
			demand_idx = numpy.array(list(numpy.linspace(0, other_vcpus - 0.08 , 3 * other_vcpus)) + [other_vcpus])
			demand_delta = demand_idx[1] - demand_idx[0]
			usage_by_other_demand = [ list() for i in range(0,len(demand_idx)) ]

			for i in range(0,len(other_demand)):
				if other_demand[i] == other_vcpus:
					od_i = len(demand_idx)-1
				else:
					od_i = int(round(other_demand[i]/demand_delta,0))
				usage_by_other_demand[od_i].append(usage[i])

			self._cache_other_demand_idx = numpy.array(demand_idx)
			self._cache_guest_usage_by_other_demand = [ numpy.array(i) if len(i) > 0 else list() for i in usage_by_other_demand ]
	@property
	def other_demand_idx(self):
		if self._cache_other_demand_idx is None:
			self._construct_other_demand_cache()
		return self._cache_other_demand_idx
	def other_demand_idx_index(self, other_demand):
		oidx = self.other_demand_idx
		i=0
		while i < len(oidx) and other_demand > oidx[i]: i += 1
		if i == 0: return 0
		if other_demand - oidx[i-1] < oidx[i] - other_demand:
			return i-1
		else:
			return i
	@property
	def guest_usage_by_other_demand(self):
		if self._cache_guest_usage_by_other_demand is None:
			self._construct_other_demand_cache()
		return self._cache_guest_usage_by_other_demand
	def usage_percentiles(self, p):
		return numpy.array([ numpy.percentile(i, p) if len(i) > 0 else None for i in self.guest_usage_by_other_demand ])
	def usage_percentiles_cut(self, p, usage):
		demand = list(self.other_demand_idx)
		percentiles = list(self.usage_percentiles(p))
		while len(demand) > 0 and percentiles[0] > usage:
			demand.pop(0)
			percentiles.pop(0)
		return (numpy.array(demand), numpy.array(percentiles))

	_cache_model = None
	@property
	def model(self):
		if self._cache_model == None:
			self._cache_model = guestmodel.GuestModel(
				self.host_cpus,
				self.total_vcpus,
				self.guest_vcpus)
		return self._cache_model

	def fitF(self, f, percentile, method=None):
		x = self.other_demand_idx
		y = self.usage_percentiles(percentile)

		if method is None and (self.host_cpus, self.total_vcpus) in [(3,5)]:
			method = 'dogbox'
		try:
			popt, _ = curve_fit(f, x, y, method=method)
		except RuntimeError:
			popt, _ = curve_fit(f, x, y, method='dogbox')
		return popt

	#sigmoid function
	@staticmethod
	def sigmoid(x, x0, y0, c, k):
		return c / (1 + numpy.exp(-k*(x-x0))) + y0

	def fitSigmoid(self, percentile):
		f = self.sigmoid
		x = self.other_demand_idx
		y = self.usage_percentiles(percentile)
		method = None if (self.host_cpus, self.total_vcpus) not in [(3,5)] else 'dogbox'

		try:
			popt, _ = curve_fit(f, x, y, method=method)
		except RuntimeError:
			try:
				popt, _ = curve_fit(f, x, y, method='dogbox')
			except RuntimeError:
				x0 = x[-1]
				f2 = lambda x, y0, c, k: f(x, x0, y0, c, k)
				popt, _ = curve_fit(f2, x, y, method='dogbox')
				popt = [x0] + list(popt)
		return (f, popt)

	def getDemandByT(self, percentile, T, f=sigmoid):
		idx = self.other_demand_idx

		if f is None:
			value_p = self.usage_percentiles(percentile)
		else:
			fit_p = self.fitF(f, percentile)
			value_p = f(idx, *fit_p)

		for i in range(0, len(idx)):
			if value_p[i] < T:
				i2 = i-1 if i > 0 else 0
				return idx[i2]
		return idx[-1]

	def fitUpper(self):
		percentile = 95
		_, maxdemand = self.model.upperMinMaxDemand
		def f(x, y0, c, k):
			return self.sigmoid(x, maxdemand, y0, c, k)
		popt = self.fitF(f, percentile)
		return (self.sigmoid, [maxdemand] + list(popt))


class ExperimentSet:
	experiments = collections.OrderedDict()

	def add(self, exp):
		self.experiments[(exp.host_cpus, exp.total_vcpus)] = exp

	def keys(self): 	  return self.experiments.keys()
	def values(self): return self.experiments.values()
	def items(self):  return self.experiments.items()

	def fitPercentileT(self, percentile, T, f=None):
		X = []
		Y = []
		for k, d in self.items():
			idx = (d.total_vcpus - d.guest_vcpus - d.getDemandByT(percentile, T, f=f))
			Y.append(idx)
			X.append([d.host_cpus, d.total_vcpus])
		lr = linear_model.LinearRegression()
		lr.fit(X, Y)
		return lr

	def fitMinDemand(self):
		X, Y = [], []
		Yp5 = []
		for k, d in self.items():
			_, maxdemand = d.model.upperMinMaxDemand
			X.append((d.host_cpus, d.total_vcpus))
			y = d.usage_percentiles(95)
			Y.append(y[-1])
			y = d.usage_percentiles(5)
			Yp5.append(y[-1])

		print('X', X)
		print('Y', Y)

		fig, ax = plt.subplots(1, 1)
		fig.set_figheight(5)
		ax.set_xlabel('N')
		ax.set_ylabel('Ti')
		ax.grid()

		colors = ['red', 'blue']
		for cpu in set([int(d[0]) for d in X]):
			color = colors.pop(0)
			X2, Y2, Y2p5 = [], [], []
			for i in range(0,len(X)):
				if X[i][0] == cpu:
					X2.append(X[i])
					Y2.append(Y[i])
					Y2p5.append(Yp5[i])
			ax.plot([x[1] for x in X2], Y2p5, 'x', label='M={} p05'.format(cpu), color=color)
			ax.plot([x[1] for x in X2], Y2, 'o', label='M={} p95'.format(cpu), color=color)

			ax.plot([x[1] for x in X2], [1. / numpy.ceil((x[1]) / (x[0])) for x in X2], ':', lw=2, label='M={} l^fin'.format(cpu), color=color)
			ax.plot([x[1] for x in X2], [1. / numpy.floor((x[1]) / (x[0])) for x in X2], '-', lw=1, label='M={} u^fin'.format(cpu), color=color)

		ax.legend(loc='best', frameon=False)
		fig.savefig('lfin,ufin.pdf')
		plt.show()


	def fitUpperS(self):
		flinear = lambda x, a, b: a*x + b
		inter = lambda f1, f2: (f1[1] - f2[1])/(f2[0] - f1[0])
		X = []
		Y = []
		for k, d in self.items():
			mindemand, maxdemand = d.model.upperMinMaxDemand
			minimum, maximum = d.model.upperMinMaxT
			fU, pU = d.fitUpper()
			midX = numpy.array([maxdemand-.01, maxdemand])
			midY = fU(midX, *pU)
			midP, _ = curve_fit(flinear, midX, midY)
			X.append([mindemand, maxdemand, minimum])
			Y.append(inter(midP, (0,maximum)))

		return linear_model.LinearRegression().fit(X, Y)

###########################################################################
###########################################################################
experiments = ExperimentSet()
for experiment_idx in config.experiments:

	experiment_data = VMStats(*experiment_idx, 1)
	experiments.add(experiment_data)

	#############################################################
	# Telemetry: Usage, Steal, and Demand
	if False:
		fig, axs = plt.subplots(experiment_data.total_vcpus, 1)
		fig.set_figheight(experiment_idx[1])

		for i in range(1, experiment_data.total_vcpus +1):
			stats = experiment_data if i == 1 else VMStats(*experiment_idx, i)
			ax = axs[i-1]
			ax.set_ylim(bottom=-0.05, top=1.1)
			ax.plot(stats.time, stats.guest_usage,  lw=1)
			ax.plot(stats.time, stats.guest_steal,  lw=1)
			ax.plot(stats.time, stats.guest_demand,  lw=1)
			if i < experiment_idx[1]:
				plt.setp(ax.get_xticklabels(), visible=False)
			else:
				ax.set_xlabel('experiment time (s)')

		fig.legend(loc='lower center', labels=['time (Ti)', 'steal time (Si)', 'demand (Di)'], ncol=3, frameon=True)
		#fig.savefig('telemetry-{},{}.pdf'.format(*experiment_idx))
		plt.show()


	experiment_data.cutThreshold(.98)


#experiments.fitMinDemand()

#############################################################
# Data and Models per experiment
def printDataModels(experiment=None, save=None):
	for experiment_idx, experiment_data in experiments.items():
		if experiment is not None and experiment != experiment_idx and experiment_idx not in experiment: continue
		fig, ax = plt.subplots(1, 1)
		fig.set_figheight(3)
		#ax.set_title('Models: CPUs {}, VMs {}'.format(experiment_data.host_cpus, experiment_data.total_vcpus))
		ax.grid()


		guest_model = experiment_data.model
		#ax.set_ylim(bottom=-5, top=110)
		ax.plot(experiment_data.other_demand, experiment_data.guest_usage,   '.', lw=1, color="#447F95", label='data')

		#ax.plot(experiment_data.other_demand_idx, [ guest_model.predict1(o) for o in experiment_data.other_demand_idx ], '-', lw=2.5, color='#07BA4F', label='initial m')

		if save in [None, 'model']:
			ax.plot(experiment_data.other_demand_idx, [ guest_model.predictLower(o) for o in experiment_data.other_demand_idx ], '-', lw=2.5, color='#14C0CC', label='lower')
			ax.plot(experiment_data.other_demand_idx, [ guest_model.predictUpper(o) for o in experiment_data.other_demand_idx ], '-', lw=2.5, color='#14C0CC', label='upper')

		#fit_f, fit_p = experiment_data.fitSigmoid(5)
		#ax.plot(experiment_data.other_demand_idx, fit_f(experiment_data.other_demand_idx, *fit_p), '-', lw=2.5, color='black')

		if True:
			ax.plot(experiment_data.other_demand_idx, experiment_data.usage_percentiles( 5), '-', lw=2.5, color='orange', label='p05')
			ax.plot(experiment_data.other_demand_idx, experiment_data.usage_percentiles(50), '-', lw=2.5, color='#EEBF67', label='p50')
			ax.plot(experiment_data.other_demand_idx, experiment_data.usage_percentiles(95), '-', lw=2.5, color='orange', label='p95')

		if False:
			def f2(x, a, b, c, e1):
				return a*numpy.log(x)**2 + b*x**e1 + c
			def f3(x, a, b, c, d):
				return (a*x + b)/(c*x + d)
			def f(x, a, b, c):
				return -(a**x + b*x + c)

			x1, y1 = experiment_data.usage_percentiles_cut(95, .98)
			popt, _ = curve_fit(f, x1, y1)
			print('fit upper:', popt)
			ax.plot(x1, f(x1, *popt), '-', lw=2.5, color='red')


		ax.set_ylabel('T1')
		ax.set_xlabel('D\'1')
		ax.legend(loc='best', ncol=2, frameon=True)
		if save == 'data':
			fig.savefig('data-{},{}.pdf'.format(*experiment_idx))
		elif save == 'model':
			fig.savefig('models-{},{}.pdf'.format(*experiment_idx))
		plt.show()


		#############################################################
		# Models minus Data
		if False:
			fig, ax = plt.subplots(1, 1)
			fig.set_figheight(3)
			ax.set_title('Upper Model - Data: CPUs {}, VMs {}'.format(experiment_data.host_cpus, experiment_data.total_vcpus))
			ax.plot(experiment_data.other_demand_idx, [0 for i in range(0,len(experiment_data.other_demand_idx))], '-', lw=1, color='black')
			ax.plot(experiment_data.other_demand, [ s.guest_usage - guest_model.predictUpper(s.total_demand - s.guest_demand) for s in experiment_data.stats ], '.', lw=1, color="#447F95", label='data')
			ax.set_ylabel('Ti - upper model')
			ax.set_xlabel('D\'i')
			ax.legend(loc='best', ncol=2, frameon=True)
			plt.show()

			fig, ax = plt.subplots(1, 1)
			fig.set_figheight(3)
			ax.set_title('Lower Model - Data: CPUs {}, VMs {}'.format(experiment_data.host_cpus, experiment_data.total_vcpus))
			ax.plot(experiment_data.other_demand_idx, [0 for i in range(0,len(experiment_data.other_demand_idx))], '-', lw=1, color='black')
			ax.plot(experiment_data.other_demand, [ s.guest_usage - guest_model.predictLower(s.total_demand - s.guest_demand) for s in experiment_data.stats ], '.', lw=1, color="#447F95", label='data')
			ax.set_ylabel('Ti - lower model')
			ax.set_xlabel('D\'i')
			ax.legend(loc='best', ncol=2, frameon=True)
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

#printDataModels((2,5))
#printDataModels((2,6))
printDataModels(save='model')

def printPotential(other_demand=None, exp=None):
	X = numpy.linspace(0,1,50)

	for eIdx, eData in experiments.items():
		if exp is not None and exp != eIdx and eIdx not in exp: continue

		od_list = [ x/2. for x in range( 0, 2*(eIdx[1]-eData.guest_vcpus) +1 )] if other_demand is None else [other_demand]

		for od in od_list:
			fig, ax = plt.subplots(1, 1)
			fig.set_figheight(2.5)
			ax.grid()
			ax.set_title('Potential CPU time M={}, N={}, D\'i={}'.format(eIdx[0], eIdx[1], od))
			ax.set_ylabel('Pr')
			ax.set_xlabel('Pi >= x')

			colors = ['C1', 'C0']

			sp = []
			f, p = eData.model.potentialSF(od, save_points=sp)
			ax.plot(X, f(X, *p), '-', label='model', color=colors.pop(0))
			if len(sp) == 2:
				ax.plot(sp[0], sp[1], '+', label='ref points')

			data = list(eData.guest_usage_by_other_demand[eData.other_demand_idx_index(od)])
			data.sort()
			Y = numpy.zeros(50)
			for i in range(len(X)-1, 0, -1):
				limit = (X[i]+X[i-1])/2.
				while len(data) > 0 and data[-1] > limit:
					Y[i] += 1
					data.pop()
			Y = Y/sum(Y)
			for i in range(len(Y)-2, -1, -1): Y[i] += Y[i+1]

			ax.plot(X, Y, '-', label='data D\'1={:.1f}'.format(od), color=colors.pop(0))

			ax.legend(loc='best', frameon=True)
			fig.savefig('potential-{},{},{}.pdf'.format(*eIdx, '{:.1f}'.format(od).replace('.','_')))
			plt.show()

#exp=(3,7)
#printDataModels(exp)
#printPotential()

#############################################################
# Summary of Experiments
def printSummary(percentile, experiment=None):
	flinear = lambda x, a, b: a*x + b
	inter = lambda f1, f2: (f1[1] - f2[1])/(f2[0] - f1[0])

	fig, ax = plt.subplots(1, 1)
	fig.set_figheight(3)
	ax.grid()
	add_title = ', M={}, N={}'.format(*experiment) if experiment is not None else ''
	#ax.set_title('Normalized Percentile {}{}'.format(percentile, add_title))
	ax.set_ylabel('Ti')
	ax.set_xlabel('D\'i')
	filename = None

	if percentile == 5:
		ret_dataT= collections.OrderedDict()
		ret_dataB= collections.OrderedDict()
	if percentile == 95:
		lrS = experiments.fitUpperS()

	for experiment_idx, experiment_data in experiments.items():
		if experiment is not None and experiment != experiment_idx and experiment_idx not in experiment: continue
		cpus, total_vcpus = experiment_idx
		guest_vcpus = experiment_data.guest_vcpus
		other_vcpus = total_vcpus - guest_vcpus

		x = experiment_data.other_demand_idx
		y = experiment_data.usage_percentiles(percentile)
		ax.plot(x, y, '-', lw=1, label='data')

		x2 = numpy.linspace(0,x[-1],50)

		if percentile == 5: # curve fit Lower
			f, popt = experiment_data.fitSigmoid(percentile)
			print(experiment_idx, 'fit ', popt)
			ax.plot(x2, f(x2, *popt), ':', lw=2, label='data fit')

			minimum, maximum = experiment_data.model.lowerMinMaxT
			mindemand, maxdemand = experiment_data.model.lowerMinMaxDemand

			print(experiment_idx, 'init-end', [mindemand, x[-1]], [maximum, minimum])
			#ax.plot([mindemand, x[-1]], [maximum, minimum], '-', label='M={}, N={} init-end'.format(cpus, total_vcpus), lw=2)

			popt_x0 = popt[0]
			x3 = numpy.array([popt_x0-.01, popt_x0+.01])
			y3 = f(x3, *popt)
			popt_mid, _ = curve_fit(flinear, x3, y3)
			#ax.plot(x2, [i if i > 0.2 and i <=1.1 else None for i in flinear(x2, *popt_mid)], '-', label='M={}, N={} mid'.format(cpus, total_vcpus), lw=2)

			x4 = [mindemand, inter(popt_mid, (0, maximum)), inter(popt_mid, (0, minimum)), maxdemand]
			y4 = [maximum, maximum, minimum, minimum]
			#ax.plot(x4, y4, 'o')
			#ax.plot([x4[1], x4[2]], [maximum, minimum], '-', label='inflect. point angle')

			ret_dataB[(experiment_idx[0], experiment_idx[1], maximum-minimum, x4[3] - x4[0])] = x4[2]/x4[3]
			ret_dataT[(cpus, total_vcpus, mindemand, maxdemand)] = x4[1]-x4[0] if x4[1]-x4[0] >= 0 else 0.

			ax.plot(x2, [experiment_data.model.predictLower(i) for i in x2], '-', label='lower function', lw=2)
			lowerPoints = experiment_data.model._cache_predictLower_points
			ax.plot(lowerPoints[0], lowerPoints[1], '+', label='ref. points', color='purple')

			filename = 'fit_lower-{},{}.pdf'.format(*experiment_idx)

		elif percentile == 95: # curve fit Upper
			mindemand, maxdemand = experiment_data.model.upperMinMaxDemand
			minimum, maximum = experiment_data.model.upperMinMaxT

			f, popt = experiment_data.fitUpper()

			print(experiment_idx, 'fit ', popt)
			ax.plot(x2, f(x2, *popt), ':', label='data fit', lw=2)

			midX = numpy.array([maxdemand-.01, maxdemand])
			midY = f(midX, *popt)
			midP, _ = curve_fit(flinear, midX, midY)
			#ax.plot([inter(midP, (0,maximum)), maxdemand], [maximum, f(maxdemand, *popt)], 'o', label='fit ref')

			ax.plot(x2, [experiment_data.model.predictUpper(i) for i in x2], '-', label='upper function')
			upperPoints = experiment_data.model._cache_predictUpper_points
			if upperPoints is not None:
				ax.plot(upperPoints[0], upperPoints[1], '+', label='ref. points', color='purple')

			filename = 'fit_upper-{},{}.pdf'.format(*experiment_idx)

	ax.legend(loc='best', frameon=False)
	if filename is not None:
		fig.savefig(filename)
	plt.show()

	if percentile == 5:
		lrT = linear_model.LinearRegression()
		lrT.fit(list(ret_dataT.keys()), list(ret_dataT.values()))
		lrB = linear_model.LinearRegression()
		lrB.fit(list(ret_dataB.keys()), list(ret_dataB.values()))
		return (lrB, lrT)
	elif percentile == 95:
		pass

if False:
	printSummary(5, (2,4))
	printSummary(5, (2,5))
	printSummary(5, (2,6))
	printSummary(5, (2,7))
	printSummary(5, (2,8))
	printSummary(5, (3,4))
	printSummary(5, (3,5))
	printSummary(5, (3,6))
	printSummary(5, (3,7))
	printSummary(5, (3,8))
	printSummary(5, (3,9))
if False:
	printSummary(95, (2,4))
	printSummary(95, (2,5))
	printSummary(95, (2,6))
	printSummary(95, (2,7))
	printSummary(95, (2,8))
	printSummary(95, (3,4))
	printSummary(95, (3,5))
	printSummary(95, (3,6))
	printSummary(95, (3,7))
	printSummary(95, (3,8))
	printSummary(95, (3,9))
	#printSummary(5, [(3,7),(3,8), (3,9)])
if False:
	lrs = printSummary(5)
	#printSummary(95)

def fitBegin(percentile, cut_threshold):
	def fL(x, a, b): #linear
		return a*x + b
	def fE(x, a, b, c): # exponential
		return -(a**x + b*x + c)
	f = None

	fig, ax = plt.subplots(1, 1)
	fig.set_figheight(3)
	#ax.set_title('Fit Initials: percentile {}, threshold {}'.format(percentile, cut_threshold))
	ax.set_ylabel('V\'i - D\'i')
	ax.set_xlabel('N')
	ax.grid()

	data_p_ini = []
	for experiment_idx, experiment_data in experiments.items():
		cpus, total_vcpus = experiment_idx
		guest_vcpus = experiment_data.guest_vcpus

		y = (total_vcpus - guest_vcpus - experiment_data.getDemandByT(percentile, cut_threshold, f=f))
		#y = experiment_data.getDemandByT(percentile, cut_threshold, f=f)/(total_vcpus - guest_vcpus)
		data_p_ini.append((cpus, total_vcpus, y))

	data_p_ini.sort(key=lambda x: x[0])
	x = numpy.array([ d[1] for d in data_p_ini ])
	y = numpy.array([ d[2] for d in data_p_ini ])
	#ax.plot( x, y,'+', label='data', color='black')

	def filter(data, cpus):
		ret = []
		for i in data:
			if i[0] == cpus:
				ret.append(i)
		return ret
	data_per_cpu = collections.OrderedDict()
	for i in set([d[0] for d in data_p_ini]):
		data_per_cpu[i] = filter(data_p_ini, i)
	ax.plot( [ d[1] for d in data_per_cpu[2] ], [ d[2] for d in data_per_cpu[2] ], '*', label='data M=2', color='red')
	ax.plot( [ d[1] for d in data_per_cpu[3] ], [ d[2] for d in data_per_cpu[3] ], '*', label='data M=3', color='blue')

	#"""
	lr = experiments.fitPercentileT(percentile, cut_threshold, f=f)

	x2 = numpy.linspace(0, 9, 50)

	xaux = numpy.array([(2, i) for i in x2])
	ax.plot(x2, util.minmax(lr.predict(xaux), 0., None), '-', label='fit M=2', color='red')
	xaux = numpy.array([(3, i) for i in x2])
	ax.plot(x2, util.minmax(lr.predict(xaux), 0., None), '-', label='fit M=3', color='blue')
	n=4
	xaux = numpy.array([(n, i) for i in x2])
	ax.plot(x2, util.minmax(lr.predict(xaux), 0., None), '.', label='projection M={}'.format(n), color='darkred')
	#"""

	ax.legend(loc='best', frameon=False)
	#fig.savefig('initials_p{}_cut{}.pdf'.format(percentile, int(cut_threshold*100)))
	plt.show()

	return lr

#p05 = fitBegin( 5,.97)
#p50 = fitBegin(50,.97)
#p95 = fitBegin(95,.97)




'''
#############################################################
# Tests
from scipy.stats import norm

"""
fig, ax = plt.subplots(1, 1)
fig.set_figheight(1.7)
#plt.setp( ax.get_xticklabels(), visible=False)
ax.set_xlabel('x')
ax.set_ylabel('Pr')

X = numpy.linspace(norm.ppf(.001), norm.ppf(.999),50)
Y = norm.sf(X, scale=.6)
ax.plot(X/6 + .5,Y)

#ax.legend(loc='best', frameon=False)
fig.savefig('survival_function.pdf')
plt.show()
"""


def fitNorm(min, max):
	delta = norm.ppf(.95) - norm.ppf(.05)

	n = norm(loc=(min+max)/2.,scale=numpy.abs(max-min)/delta)
	X = numpy.linspace(n.ppf(.01), n.ppf(.99),50)
	Y = n.sf(X)
	plt.grid()
	plt.plot(X,Y)

	return n






#'''
