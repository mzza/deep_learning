import get_data
import network
get_data.load_data('mnist.pkl')
train_data,validation_data,test_data=get_data.load_data('mnist.pkl')

clf=network.Network(train_data[0].shape[1],20,10,1,'mini_batch')
clf.fit(train_data[0],train_data[1])   
accurate_rate=clf.predict(test_data[0],test_data[1])
