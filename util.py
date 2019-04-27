#!/usr/bin/env python3

import mmap

def max_round(value, max, decimals):
    return max if value > max else round(value, decimals)

class LogFile:
	lines = 0
	max_lines = \
	min_lines = None

	file = \
	mmap = None
	init_size = 5

	def __init__(self, name, min_lines, max_lines):
		assert isinstance(max_lines, int) and max_lines > 5
		assert isinstance(min_lines, int) and min_lines > 5 and max_lines >= min_lines
		self.max_lines = max_lines
		self.min_lines = min_lines

		self.file = open(name, 'w+')
		assert self.file != None, 'unable to open the file'
		self.file.truncate(self.init_size)
		self.mmap = mmap.mmap(self.file.fileno(), 0)
		self.mmap.seek(0)

	def __del__(self):
		self.mmap.close()
		self.file.close()

	def writeLine(self, line):
		mm = self.mmap
		reduced = 0
		if self.lines > self.max_lines:
			reduced = self.truncate()
			mm.seek( mm.size() - reduced )

		b = line.encode()
		self.lines += 1

		mm.resize( mm.size() - self.init_size - reduced + len(b) )
		mm.write(b)
		mm.flush()
		self.init_size = 0

	def truncate(self):
		mm = self.mmap

		mm.seek(0)
		size = 0
		for i in range(0, self.lines - self.min_lines):
			size += len(mm.readline())

		mm.move(0, size, mm.size()-size)
		self.lines = self.min_lines

		# reduced size in bytes
		return size
