# -*- coding: utf-8 -*-
'''
	Justin Chen
	session.py

	4.30.2018

	Tool for examining session objects in the parameter server
'''

from __future__ import division
import redis, ujson, argparse, sys, os, subprocess
from pprint import pprint

class ToolBox(object):
	def __init__(self, args):
		self.cache = redis.StrictRedis(host=args.host, port=args.port, db=args.db)


	'''
	'''
	def keys(self):
		pprint('keys: {}'.format([key for key in self.cache.scan_iter("*")]))


	'''
	'''
	def get_object(self, var):
		if not self.cache.exists(var):
			return 'key dne: {}'.format(var)
		else:
			return ujson.loads(self.cache.get(var))


	'''
	Clear all logs
	'''
	def clear(self, log_dir):
		for file in os.listdir(log_dir):
			if file.endswith('.log'): os.remove(os.path.join(log_dir, file))


	'''
	Display session object memory size
	'''
	def size(self, sess):
		print('{} (bytes)'.format(sys.getsizeof(sess)))


	'''
	Print files

	Input: files (list) List of files
	'''
	def print_files(self, files, title='files'):
		# Print incomplete hyperedges
		div = '-'*(len(title)+1)
		print '\n{}\n{}:\n{}'.format(div, title, div)
		for i in files:
			if isinstance(i,tuple):
				print '{}\n⤷{}\n'.format(*i)
			else:
				print i


	'''
	Aggregate logs from peers
	'''
	def pull_logs(self):
		os.system('chmod +x ./logs/pull.sh')
		log_dir = os.path.join(os.getcwd(), 'logs', 'pull.sh')
		subprocess.call(log_dir, shell=True)


	'''
	Return all log files

	Input:  log_dir (string) Path to directory containing logs
	Output: logs    (list)   List of all log files
	'''
	def get_logs(self, log_dir):
		return [os.path.join(log_dir, file) for file in os.listdir(log_dir) if file.endswith('.log')]


	'''
	Find all files containing given term

	Input:  paths (list) List of file paths
		    debug (bool) Set to True to grab last function call in logs
	Output: count (int)  Number of files containing term
	        files (list) List of file paths
	'''
	def grep_all(self, term, paths, case=False, debug=False):
		files = {'match': [], 'mismatch': []}
		count = 0

		for log in paths:
			with open(log, 'rb') as f:
				log = log.split('/')[-1]
				content = f.read()
				match = False

				if case:
					match = term.lower() in content.lower()
				else:
					match = term in content

				if match:
					count += 1
					files['match'].append(log)
				else:
					if debug:
						# Get last function call in file
						f.seek(-2, os.SEEK_END)
						try:
							while f.read(1) != b"\n":
								f.seek(-2, os.SEEK_CUR)
							last = [s for s in f.readline().split(' ') if len(s) > 0]
							last.pop(1)
							last = '  '.join(last[:2])

							files['mismatch'].append((log, last))
						except IOError as e:
							print 'IOError: {}'.format(log)
					else:
						files['mismatch'].append(log)

		return count, files


	'''
	Check if all the sessions completed training

	Input: log_dir (string) Path to log directory
		   pull    (bool)   True if should pull logs
	'''
	def check_logs(self, log_dir, pull=False):
		if pull:
			self.pull_logs()

		while len(os.listdir(log_dir)) == 0:
			sleep(0.5)

		all_logs = []
		ps_logs = []
		for l in self.get_logs(log_dir):
			if 'ps' not in l:
				all_logs.append(l)
			else:
				ps_logs.append(l)

		total = len(all_logs)

		if total > 0:
			complete, files = self.grep_all('hyperedge complete', all_logs, debug=True)
			self.print_files(files['mismatch'], 'incomplete hyperedges')
			self.print_files(files['match'], 'completed')
			print('completed hyperedges: {}/{} ({}%)'.format(complete, total, 100*complete/total))

			total = len(ps_logs)
			complete, files = self.grep_all('Hypergraph Complete', ps_logs)
			self.print_files(files['mismatch'], 'incomplete peers')
			print('completed training: {}/{} ({}%)'.format(complete, total, 100*complete/total))

			self.print_files(self.get_edges(), 'variables')
		else:
			print('no logs. rerun experiment.')


	def get_edges(self):
		return ['current edges:   {}'.format(self.get_object('curr_edges')),
		        'hyperedge count: {}'.format(self.get_object('hyperedges'))]


	'''
	Query session objects
	'''
	def query(self, sess, args):
		result = sess
		if args.property != None:
			result = 'key:{}, value: {}'.format(args.property, sess[args.property])
		elif args.properties:
			result = 'properties: {}'.format(list(sess.keys()))
		elif args.ignore != None:
			for k in args.ignore:
				if k in sess:
					del sess[k]
			result = sess
		elif args.minimal:
			for k in ['parameters', 'gradients', 'multistep']:
				if k in sess:
					del sess[k]
			result = sess
		pprint(result)


def main():
	parser = argparse.ArgumentParser()
	parser.add_argument('--host', type=str, default='localhost', help='Redis host')
	parser.add_argument('--port', type=int, default=6379, help='Redis port')
	parser.add_argument('--case', '-c', action='store_true', help='Set for case sensitive matching when using the --grep option')
	parser.add_argument('--check', '-ch', action='store_true', help='Check that all hyperedges completed training')
	parser.add_argument('--clear', action='store_true', help='Clear all logs')
	parser.add_argument('--db', type=int, default=0, help='Redis db')
	parser.add_argument('--edges', '-e', action='store_true', help='Display edge count')
	parser.add_argument('--grep', '-g', type=str, help='Grep all files for given term')
	parser.add_argument('--ignore', '-i', type=str, nargs='+', help='Ignores a particular key/value in the session object')
	parser.add_argument('--keys', '-k', action='store_true', help='Get all Redis keys')
	parser.add_argument('--log_dir', '-l', type=str, default=os.path.join(os.getcwd(), 'logs'), help='Log directory')
	parser.add_argument('--minimal', '-m', action='store_true', help='Ignore parameters and gradients')
	parser.add_argument('--sess', '-s', type=str, help='Session objection id')
	parser.add_argument('--size', '-z', action='store_true', help='Get size of cache object')
	parser.add_argument('--property', '-p', type=str, help='Session object property')
	parser.add_argument('--properties', '-ps', action='store_true', help='Get all properties of object')
	parser.add_argument('--pull', '-pl', action='store_true', help='Pull logs')
	parser.add_argument('--variable', '-v', type=str, help='Retrieve state variable. If using this, do not set --sess')
	args = parser.parse_args()

	tb = ToolBox(args)

	# Clear logs
	if args.clear:
		tb.clear(args.log_dir)

	if args.check:
		tb.check_logs(args.log_dir, pull=args.pull)

	if args.edges:
		tb.print_files(tb.get_edges(), 'variables')

	if args.grep != None:
		all_logs = tb.get_logs(args.log_dir)
		total = len(all_logs)
		count, files = tb.grep_all(args.grep, all_logs, case=args.case)
		tb.print_files(files, 'matching files')
		print('matching files: {}/{} ({}%)'.format(count, total, 100*count/total))
	
	# Display all available keys
	if args.keys:
		tb.keys()

	sess = ''
	if args.sess != None:
		sess = tb.get_object(args.sess)
		if len(sess) > 0:
			# Return size of session object
			if args.size:
				tb.size(sess)

			# Query session objects and variables
			tb.query(sess, args)

	elif args.variable != None:
		print tb.get_object(args.variable)


if __name__ == '__main__':
	main()