import sys
import os
from typing import Tuple
from numpy.core.fromnumeric import trace

from numpy.random.mtrand import random


sys.path.append('C:\\Users\\coton\\Desktop\\github\\fema\\src\\')


import numpy as np
import math 
from sklearn.metrics.pairwise import euclidean_distances

from fem_basis import Basis    


class FEMaFeatureLearning:
    """
    Class responsible to learn the main features for classification
    problems
    """
    INTER = 0
    INTRA = 1

    def __init__(self) -> None:
        self.train_x = None
        self.train_y = None
        self.eval_x = None
        self.eval_y = None
        self.num_classes = 0
        self.num_features = 0
        self.features_weigth = []
        self.basis = None
        self.k = 0

    def __init__(self, k:int=2, basis=Basis.shepardBasis) -> None:
        self.train_x = None
        self.train_y = None
        self.eval_x = None
        self.eval_y = None
        self.num_classes = 0
        self.num_features = 0
        self.features_weigth = []
        self.k = k
        self.basis = basis

    def fit(self, train_x:np.array, train_y:np.array, eval_x:np.array, eval_y:np.array) -> np.array:
        """Method responsible to create the manifold learning probabilities

        Args:
            train_x (np.array): Features of the training set
            train_y (np.array): Class of the training set
            eval_x (np.array): Features of the validation set
            eval_y (np.array): Class of the validation set
        """
        self.train_x = train_x
        self.train_y = train_y
        self.eval_x = eval_x
        self.eval_y = eval_y
        self.num_train_samples = len(train_y)
        self.num_features = train_x.shape[1]
        self.num_classes = len(set(train_y[:,0]))
        self.num_eval_samples = len(eval_y)
        #One weigth for Intra and another for Inter
        self.features_weigth = np.zeros((2,self.num_classes, self.num_features))

        
        #TODO: We need improve the model considering the matrix of classes instead class vector       
        #TODO: Insert the manifold probabilistic in the model


        for c in range(self.num_classes):                        
            
            mask_inter = (self.train_y[:,0] != c)
            mask_intra = (self.train_y[:,0] == c)
            
            

            for f in range(self.num_features):                            
                features_inter = train_x[mask_inter][:,f]                
                features_intra = train_x[mask_intra][:,f]

                features_inter = features_inter.reshape(-1,1)
                features_intra = features_intra.reshape(-1,1)

                dist_intra = euclidean_distances(features_intra,features_intra).mean()
                dist_inter = euclidean_distances(features_inter,features_intra).mean()

                self.features_weigth[self.INTER ,c, f] = dist_inter
                self.features_weigth[self.INTRA ,c, f] = dist_intra

        return self.features_weigth            

