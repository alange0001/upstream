#!/usr/bin/env python3

import collections
import json

def max_round(value, max, decimals):
	return max if value > max else round(value, decimals)

def coalesce(*args):
	for i in args:
		if i != None:
			return i
	return None

class LogFile:
	_file = \
	_format = None
	_close_file = True

	def __init__(self, file, format):
		assert format in ['csv', 'json']
		self._format = format
		if isinstance(file, str):
			self._file = open(file, 'w')
			assert self._file != None, 'unable to open the file "{}"'.format(file)
		else:
			self._file = file
			self._close_file = False

	def __del__(self):
		if self._close_file:
			self._file.close()

	def write(self, data):
		self._file.write(self._formatLine(data))
		self._file.flush()

	_firstLine = True
	def _formatLine(self, data):
		if self._format == 'json':
			return '{}\n'.format(json.dumps(data))
		elif self._format == 'csv':
			ret = ''
			if self._firstLine:
				self._firstLine = False
				ret = '#{}\n'.format('; '.join(self._getCSVHeader(None, data)))
			ret += '{}\n'.format('; '.join(self._getCSVLine(data)))
			return ret

	def _getCSVHeader(self, prefix, data):
		ret = []
		if isinstance(data, list) or isinstance(data, tuple):
			for i in range(0, len(data)):
				prefix_i = '{}_{}'.format(prefix, i) if prefix != None else i
				ret_i = self._getCSVHeader(prefix_i, data[i])
				ret.extend(ret_i)
		elif isinstance(data, dict) or isinstance(data, collections.OrderedDict):
			for k, v in data.items():
				prefix_i = '{}_{}'.format(prefix, k) if prefix != None else k
				ret_i = self._getCSVHeader(prefix_i, v)
				ret.extend(ret_i)
		else:
			return [prefix]
		return ret

	def _getCSVLine(self, data):
		ret = []
		if isinstance(data, list) or isinstance(data, tuple):
			for v in data:
				ret.extend(self._getCSVLine(v))
		elif isinstance(data, dict) or isinstance(data, collections.OrderedDict):
			for v in data.values():
				ret.extend(self._getCSVLine(v))
		else:
			return [str(data)]
		return ret

class WatchLog:
	_file        = None
	_format      = None
	_file_name   = None
	_header_size = 0
	_header      = []
	_lines       = []

	def __init__(self, file, format):
		assert format in ['csv', 'json']
		self._format = format

		self._file_name = file
		self._file = open(file, 'r')
		assert self._file != None, 'unable to open the file "{}"'.format(file)

	def __del__(self):
		if self._file != None:
			self._file.close()

	def read(self):
		ret = []
		lines = self._file.readlines()
		if self._format == 'csv':
			if len(self._header) == 0 and lines[0][0] == '#':
				self._header = lines[0].replace('\n', '').split('; ')
				self._header[0] = self._header[0].replace('#', '')
				self._header_size = len(self._header)
				del lines[0]
			for i in lines:
				line_list = i.replace('\n', '').split('; ')
				ret_i  = collections.OrderedDict()
				self._lines.append(ret_i)
				ret.append(ret_i)
				for j in range(0, self._header_size):
					ret_i[self._header[j]] = line_list[j]
		else: # json
			for i in lines:
				i = i.replace('\n', '')
				ret_i = json.loads(i)
				self._lines.append(ret_i)
				ret.append(ret_i)
		return ret

	def lines(self):
		return self._lines
	def header(self):
		return self._header
