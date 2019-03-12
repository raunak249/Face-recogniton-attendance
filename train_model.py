#import the necessary libraries

from sklearn.preprocessing import LabelEncoder
import argparse
import pickle
from sklearn.svm import SVC

#Declaring the argument parser and all the arguments
ap = argparse.ArgumentParser()
ap.add_argument('-e','--embeddings',default='output/embeddings.pickle',help='Path to the serialized embeddings')
ap.add_argument('-r','--recognizer',default='output/recognizer.pickle',help='Path to the trained model')
ap.add_argument('-l','--le',default='output/le.pickle',help='Path to the output label encoder')
args = vars(ap.parse_args())

print('[INFO] loading the embeddings..')
data = pickle.loads(open(args['embeddings'],'rb').read())

#initializing the label encoder
le = LabelEncoder()
labels = le.fit_transform(data['names'])

print('[INFO] initializing the model')
recognizer = SVC(C=1.0,kernel = 'linear',probability=True)
recognizer.fit(data['embeddings'],labels)

f = open(args['recognizer'],'wb')
f.write(pickle.dumps(recognizer))
f.close()

f = open(args['le'],'wb')
f.write(pickle.dumps(le))
f.close()
