'''
	Justin Chen
	session.py

	4.30.2018

	Tool for examining session objects in the parameter server
'''


import redis, ujson, argparse, sys
from pprint import pprint


def get_object(var, cache):
	if not cache.exists(var):
		return 'key dne: {}'.format(var)
	else:
		return ujson.loads(cache.get(var))


def main():
	parser = argparse.ArgumentParser()
	parser.add_argument('--host', type=str, default='localhost', help='Redis host')
	parser.add_argument('--port', type=int, default=6379, help='Redis port')
	parser.add_argument('--db', type=int, default=0, help='Redis db')
	parser.add_argument('--ignore', '-i', type=str, nargs='+', help='Ignores a particular key/value in the session object')
	parser.add_argument('--keys', '-k', action='store_true', help='Get all Redis keys')
	parser.add_argument('--minimal', '-m', action='store_true', help='Ignore parameters and gradients')
	parser.add_argument('--sess', '-s', type=str, help='Session objection id')
	parser.add_argument('--size', '-z', action='store_true', help='Get size of cache object')
	parser.add_argument('--property', '-p', type=str, help='Session object property')
	parser.add_argument('--properties', '-ps', action='store_true', help='Get all properties of object')
	parser.add_argument('--variable', '-v', type=str, help='Retrieve state variable. If using this, do not set --sess')
	args = parser.parse_args()

	cache = redis.StrictRedis(host=args.host, port=args.port, db=args.db)
	
	if args.keys:
		pprint('keys: {}'.format([key for key in cache.scan_iter("*")]))

	sess = get_object(args.sess, cache) if args.sess != None else get_object(args.variable, cache)

	if args.size:
		print('{} (bytes)'.format(sys.getsizeof(sess)))

	if args.ignore != None:
		for k in args.ignore:
			if k in sess:
				del sess[k]
		pprint(sess)
	elif args.minimal:
		for k in ['parameters', 'gradients']:
			if k in sess:
				del sess[k]
		pprint(sess)
	else:
		result = sess
		if args.property != None:
			result = 'key:{}, value: {}'.format(args.property, sess[args.property])
		if args.properties:
			result = 'properties: {}'.format(list(sess.keys()))
		pprint(result)


if __name__ == '__main__':
	main()