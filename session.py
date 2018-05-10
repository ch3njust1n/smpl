'''
	Justin Chen
	session.py

	4.30.2018

	Tool for examining session objects in the parameter server
'''

from __future__ import division
import redis, ujson, argparse, sys, os, subprocess
from pprint import pprint


def get_object(var, cache):
	if not cache.exists(var):
		return 'key dne: {}'.format(var)
	else:
		return ujson.loads(cache.get(var))

'''
Clear all logs
'''
def clear(log_dir):
	for file in os.listdir(log_dir):
		if file.endswith('.log'): os.remove(os.path.join(log_dir, file))


'''
Display session object memory size
'''
def size(sess):
	print('{} (bytes)'.format(sys.getsizeof(sess)))


'''
Print files

Input: files (list) List of files
'''
def print_files(files, title='files'):
	# Print incomplete hyperedges
	print '\n-----------\n{}:'.format(title)
	for i in files:
		print i
	print '-----------'


'''
Aggregate logs from peers
'''
def pull_logs():
	os.system('chmod +x ./logs/pull.sh')
	log_dir = os.path.join(os.getcwd(), 'logs', 'pull.sh')
	subprocess.call(log_dir, shell=True)


'''
Return all log files

Input:  log_dir (string) Path to directory containing logs
Output: logs    (list)   List of all log files
'''
def get_logs(log_dir):
	return [os.path.join(log_dir, file) for file in os.listdir(log_dir) if file.endswith('.log') and 'ps' not in file]


'''
Find all files containing given term

Input:  paths (list) List of file paths
	    match (bool) If True, return list of files containing term
	    			 Default: False
Output: count (int)  Number of files containing term
        files (list) List of file paths
'''
def grep_all(term, paths, match=True):
	files = []
	count = 0

	for log in paths:
		with open(log, 'r') as f:
			log = log.split('/')[-1]

			if term in f.read():
				count += 1
				if match:
					files.append(log)
			elif not match:
				files.append(log)

	return count, files


'''
Check if all the sessions completed training
'''
def check_logs(log_dir):
	pull_logs()

	while len(os.listdir(log_dir)) == 0:
		sleep(0.5)

	all_logs = get_logs(log_dir)
	total = len(all_logs)
	complete, incomplete = grep_all('hyperedge training complete', all_logs, match=False)

	print_files(incomplete, 'incomplete')

	print('completed hyperedges: {}/{} ({}%)'.format(complete, total, 100*complete/total))


'''
Query session objects
'''
def query(sess, args):
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
		for k in ['parameters', 'gradients']:
			if k in sess:
				del sess[k]
		result = sess
	pprint(result)


def main():
	parser = argparse.ArgumentParser()
	parser.add_argument('--host', type=str, default='localhost', help='Redis host')
	parser.add_argument('--port', type=int, default=6379, help='Redis port')
	parser.add_argument('--check', '-ch', action='store_true', help='Check that all hyperedges completed training')
	parser.add_argument('--clear', action='store_true', help='Clear all logs')
	parser.add_argument('--db', type=int, default=0, help='Redis db')
	parser.add_argument('--grep', '-g', type=str, help='Grep all files for given term')
	parser.add_argument('--ignore', '-i', type=str, nargs='+', help='Ignores a particular key/value in the session object')
	parser.add_argument('--keys', '-k', action='store_true', help='Get all Redis keys')
	parser.add_argument('--log_dir', '-l', type=str, default=os.path.join(os.getcwd(), 'logs'), help='Log directory')
	parser.add_argument('--minimal', '-m', action='store_true', help='Ignore parameters and gradients')
	parser.add_argument('--sess', '-s', type=str, help='Session objection id')
	parser.add_argument('--size', '-z', action='store_true', help='Get size of cache object')
	parser.add_argument('--property', '-p', type=str, help='Session object property')
	parser.add_argument('--properties', '-ps', action='store_true', help='Get all properties of object')
	parser.add_argument('--variable', '-v', type=str, help='Retrieve state variable. If using this, do not set --sess')
	args = parser.parse_args()

	cache = redis.StrictRedis(host=args.host, port=args.port, db=args.db)

	# Clear logs
	if args.clear:
		clear(args.log_dir)

	if args.check:
		check_logs(args.log_dir)

	if args.grep != None:
		all_logs = get_logs(args.log_dir)
		total = len(all_logs)
		count, files = grep_all(args.grep, all_logs)
		print_files(files, 'matching files')
		print('matching files: {}/{} ({}%)'.format(count, total, 100*count/total))
	
	# Display all available keys
	if args.keys:
		pprint('keys: {}'.format([key for key in cache.scan_iter("*")]))

	sess = ''
	if args.sess != None:
		sess = get_object(args.sess, cache)
	elif args.variable != None:
		sess = get_object(args.variable, cache)

	if len(sess) > 0:
		# Return size of session object
		if args.size:
			size(sess)

		# Query session objects and variables
		query(sess, args)


if __name__ == '__main__':
	main()