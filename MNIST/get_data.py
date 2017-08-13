import cPickle as pickle

def load_data(path):
	fp=open(path)
	data=pickle.load(fp)
	fp.close()
	return data

if __name__ == '__main__':
	data=load_data('mnist.pkl')
