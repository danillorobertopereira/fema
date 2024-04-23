import sys
import os
from typing import Tuple

from numpy.random.mtrand import random


#sys.path.append('C:\\Users\\coton\\Desktop\\github\\fema\\src\\')


import numpy as np
import math 

from fem_basis import Basis    

   
class FEMaSemiSupervisedClassifier:
    """
    Class responsible to perform the classification using FEMa approach
    """
    def __init__(self):
        self.train_x = None
        self.train_y = None
        self.uknw_x = None
        self.num_train_samples = 0
        self.num_uknw_samples = 0
        self.num_features = 0
        self.num_classes = 0
        self.k = 1
        self.basis = None
        self.probability_class = None
        

    def __init__(self, k:int=2, basis=Basis.shepardBasis) -> None:
        """Constructor that receives the parameter to run the FEMa

        Args:
            k (int): Define the number of neighboor used to interpolate
            basis = The finite element basis
        """
        self.train_x = None
        self.train_y = None
        self.uknw_x = None
        self.num_train_samples = 0
        self.num_uknw_samples = 0
        self.num_features = 0
        self.num_classes = 0
        self.k = k
        self.basis = basis
        self.probability_classes = None

    def fit(self, train_x:np.array, train_y:np.array, uknw_x:np.array, *args) -> None:
        """AI is creating summary for fit

        Args:
            train_x (np.array): [description]
            train_y (np.array): [description]
            uknw_x (np.array): [description]
            uknw_y (np.array, optional): [description]. Defaults to None.
        """
        self.train_x = train_x
        self.train_y = train_y
        self.uknw_x = uknw_x
        self.num_train_samples = len(train_y)
        self.num_uknw_samples = uknw_x.shape[0]
        self.num_features = self.train_x.shape[1]
        self.num_classes = len(set(train_y[:,0]))

        self.probability_classes = np.zeros((self.num_classes,self.num_train_samples))

        for i in range(self.num_classes):
            self.probability_classes[i,:] = train_y[:,0] == i

        print(self.num_uknw_samples, self.num_classes)
        
        self.uknw_yhat = np.zeros(self.num_uknw_samples)
        self.uknw_confidence_level = np.zeros((self.num_uknw_samples, self.num_classes))

        for i in range(self.num_uknw_samples):
            self.uknw_confidence_level[i,:] = [self.basis(train_x=self.train_x, train_y=self.probability_classes[c], test_one_sample=self.uknw_x[i], k=self.k, z=args[0]) for c in range(self.num_classes)]
            #self.uknw_confidence_level[i,:] = [self.basis(train_x=self.train_x, train_y=self.probability_classes[0], test_one_sample=self.uknw_x[i], k=self.k, z=args[0]) for c in range(self.num_classes)]
            self.uknw_yhat[i] = np.argmax(self.uknw_confidence_level[i,:])

        #To avoid lost the original confidence level after round or play    
        self.uknw_confidence_level_ori = self.uknw_confidence_level.copy()


    def playProbabilities(self) -> None:
        """
        Assigns confidence level probabilities 1 to unknown samples for majority class.

        This method iterates over the unknown samples and sets the confidence level of each predicted class to 1.

        Returns:
        None
        """
        self.uknw_confidence_level = self.uknw_confidence_level_ori.copy()
        u,_ = np.unique(self.uknw_yhat)
        print(u)
        for i in range(self.num_uknw_samples):
            self.uknw_confidence_level[i,int(self.uknw_yhat[i])] = 1
        return    


    def roundProbabilities(self) -> None:
        """
        Round confidence level probabilities to unknown samples
        Returns:
        None
        """
        self.uknw_confidence_level = np.rint(self.uknw_confidence_level_ori)
        return    


    def predict(self, test_x:np.array, *args) -> Tuple[np.array, np.array]:
        """AI is creating summary for predict

        Args:
            test_x (np.array): [description]

        Returns:
            Tuple[np.array, np.array]: [description]
        """

        #print('X:',self.train_x.shape, self.uknw_x.shape)
        #print('Prob and Conf',self.probability_classes.shape,self.uknw_confidence_level.transpose().shape)
        new_train_x = np.append(self.train_x, self.uknw_x,axis=0)
        #print('New Train X',new_train_x.shape)       
        new_probability_classes = np.append(self.probability_classes,self.uknw_confidence_level.transpose(),axis=1)
        #print(new_probability_classes.shape)
        num_test_samples = len(test_x)
        labels = np.zeros(num_test_samples)
        confidence_level = np.zeros((num_test_samples, self.num_classes))

        for i in range(num_test_samples):
            confidence_level[i,:] = [self.basis(train_x=new_train_x, train_y=new_probability_classes[c], test_one_sample=test_x[i], k=self.k, z=args[0]) for c in range(self.num_classes)]
            labels[i] = np.argmax(confidence_level[i,:])

        return labels, confidence_level

        
