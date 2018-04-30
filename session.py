'''
	Justin Chen
	session.py

	Tool for examining session objects
'''


import redis, ujson, argparse


def main():
	parser = argparse.ArgumentParser()
	parser.add_argument('--host', type=str, default='localhost', help='Redis host')
	parser.add_argument('--port', type=int, default=6379, help='Redis port')
	parser.add_argument('--db', type=int, default=0, help='Redis db')
	parser.add_argument('--sess', '-s', type=str, required=True, help='Session objection id')
	parser.add_argument('--key', '-k', type=str, help='Session object property')
	parser.add_argument('--properties', '-p', action='store_true', help='Get all properties of object')
	args = parser.parse_args()

	cache = redis.StrictRedis(host=args.host, port=args.port, db=args.db)

	if not cache.exists(args.sess):
		print('dne sess_id: {}'.format(args.sess))
		print([key for key in cache.scan_iter("*")])

	else:
		sess = ujson.loads(cache.get(args.sess))

		if args.key != None:
			value = sess[args.key]
			print('key:{}, value: {}'.format(args.key, value))
		if args.properties:
			print('properties: {}'.format(list(sess.keys())))



if __name__ == '__main__':
	main()