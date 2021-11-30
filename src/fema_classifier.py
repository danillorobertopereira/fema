import sys
import os
from typing import Tuple

from numpy.random.mtrand import random


sys.path.append('C:\\Users\\coton\\Desktop\\github\\fema\\src\\')


import numpy as np
import math 

from fem_basis import Basis    

   
class FEMaClassifier:
    """
    Class responsible to perform the classification using FEMa approach
    """
    def __init__(self):
        self.train_x = None
        self.train_y = None
        self.num_train_samples = 0
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
        self.num_train_samples = 0
        self.num_features = 0
        self.num_classes = 0
        self.k = k
        self.basis = basis
        self.probability_classes = None

    def fit(self, train_x:np.array, train_y:np.array) -> None:
        """Method responsible to create the manifold learning probabilities

        Args:
            train_x (np.array): Features of the training set
            train_y (np.array): Class of the training set
        """

        #TODO: Put the probability manifold learning concept 
        self.train_x = train_x
        self.train_y = train_y
        self.num_train_samples = len(train_y)
        self.num_features = self.train_x.shape[1]
        self.num_classes = len(set(train_y[:,0]))

        self.probability_classes = np.zeros((self.num_classes,self.num_train_samples))

        for i in range(self.num_classes):
            self.probability_classes[i,:] = train_y[:,0] == i

        
            
    def predict(self, test_x:np.array, *args) -> Tuple[np.array, np.array]:
        """Returns the prediction of the test set and store the

        Args:
            test_x (np.array): Features of the test set
            *args : List with the specific paramter of the Basis

        Returns:
            np.array: Prediction of the test set
        """    
        num_test_samples = len(test_x)
        labels = np.zeros(num_test_samples)
        confidence_level = np.zeros((num_test_samples, self.num_classes))

        for i in range(num_test_samples):
            confidence_level[i,:] = [self.basis(train_x=self.train_x, train_y=self.probability_classes[c], test_one_sample=test_x[i], k=self.k, z=args[0]) for c in range(self.num_classes)]
            labels[i] = np.argmax(confidence_level[i,:])

        return labels, confidence_level

    def FEMaRelax(self, test_x:np.array, k_relax:int, num_repeats:int) -> np.array:
        train_x_cp = self.train_x.copy()
        test_x_cp = test_x.copy() 

        for r in range(num_repeats):
            for i in range(self.num_train_samples):
                dist_train = np.array(
                        [
                            np.linalg.norm(train_x_cp[j]-train_x_cp[i]) for j in range(len(train_x_cp))
                        ]
                    )
                
                index_k_relax_train = np.argsort(dist_train)[1:k_relax+1]                
                signal_train = [+1 if self.train_y[i] == self.train_y[index_k_relax_train[j]] else -1  for j in range(len(index_k_relax_train))]
                
                train_x_cp[i] = (train_x_cp[index_k_relax_train]*signal_train)/len(index_k_relax_train)

            for i in range(len(test_x)):
                dist_test = np.array(
                        [
                            np.linalg.norm(train_x_cp[j]-test_x_cp[i]) for j in range(len(train_x_cp))
                        ]
                    )
                
                index_k_relax_test = np.argsort(dist_test)[1:k_relax+1]                
                
                test_x_cp[i] = np.mean(train_x_cp[index_k_relax_train],axis=0)
                

        return train_x_cp, test_x_cp




        
