#import the necessary libraries

from sklearn.preprocessing import LabelEncoder
import argparse
import pickle
from sklearn.svm import SVC

#Declaring the argument parser and all the arguments
ap = argparse.ArgumentParser()
ap.add_argument('-e','--embeddings',required=True,help='Path to the serialized embeddings')
ap.add_argument('-r','--recognizer',required=True,help='Path to the trained model')
ap.add_argument('-l','le',required=True,help='Path to the output label encoder')
args = vars(ap.parse_args())
