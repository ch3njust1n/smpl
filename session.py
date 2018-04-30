'''
	Justin Chen
	session.py

	Tool for examining session objects
'''


import redis, ujson, argparse
from pprint import pprint


def main():
	parser = argparse.ArgumentParser()
	parser.add_argument('--host', type=str, default='localhost', help='Redis host')
	parser.add_argument('--port', type=int, default=6379, help='Redis port')
	parser.add_argument('--db', type=int, default=0, help='Redis db')
	parser.add_argument('--ignore', '-i', type=str, nargs='+', help='Ignores a particular key/value in the session object')
	parser.add_argument('--keys', '-k', action='store_true', help='Get all Redis keys')
	parser.add_argument('--sess', '-s', type=str, help='Session objection id')
	parser.add_argument('--property', '-p', type=str, help='Session object property')
	parser.add_argument('--properties', '-ps', action='store_true', help='Get all properties of object')
	args = parser.parse_args()

	cache = redis.StrictRedis(host=args.host, port=args.port, db=args.db)

	
	if args.keys:
		result = 'keys: {}'.format([key for key in cache.scan_iter("*")])

	if not cache.exists(args.sess):
		result = 'dne sess_id: {}'.format(args.sess)
	else:
		sess = ujson.loads(cache.get(args.sess))

		if args.ignore != None:
			tmp = dict(sess)
			for k in args.ignore:
				del tmp[k]
			result = tmp
		else:
			if args.property != None:
				value = sess[args.property]
				result = 'key:{}, value: {}'.format(args.property, value)
			if args.properties:
				result = 'properties: {}'.format(list(sess.keys()))
	pprint(result)



if __name__ == '__main__':
	main()