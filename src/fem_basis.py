import numpy as np
import math 


class Basis:
    def __init__(self) -> None:
        pass

    def shepardBasis(train_x:np.array, train_y:np.array, test_one_sample:np.array, k:int, z:int ) -> float:
        """Compute the weigths of the interpolation considering the Shepard 
        basis and return the predicted value.

        Args:
            train_x (np.array): The feature of the training set
            train_y (np.array): The target of the training set            
            test_one_sample (np.array): One test samples
            k (int): Define the number of neighboor used to interpolate
            z (int): The power of the distance

        Returns:
            np.array: return the weigths of the training samples
        """
        dist = np.array(
            [np.linalg.norm(train_x[i]-test_one_sample) for i in range(len(train_x))]
            )
        
        mask = np.ones(len(train_x),dtype=bool)
        
        if k != 0:
            mask[np.argsort(dist)[k:]] = False

        
        dist = 1.0/(dist**z)
        weitghs = dist[mask]/sum(dist[mask]+0.0000000001)


        predicted = np.sum(weitghs*train_y[mask])

        print(predicted.shape)
        
        if math.isnan(predicted):
            predicted = np.mean(train_y)

        return predicted


    def radialBasis(train_x:np.array, train_y:np.array, test_one_sample:np.array, k:int, z:int ) -> float:
        """Compute the weigths of the interpolation considering the Shepard 
        basis and return the predicted value

        Args:
            train_x (np.array): The feature of the training set
            train_y (np.array): The target of the training set            
            test_one_sample (np.array): One test samples
            k (int): Define the number of neighboor used to interpolate
            z (int): The power of the distance

        Returns:
            np.array: return the weigths of the training samples
        """
        dist = np.array(
            [np.linalg.norm(train_x[i]-test_one_sample) for i in range(len(train_x))]
            )
        
        mask = np.ones(len(train_x),dtype=bool)
        
        if k != 0:
            mask[np.argsort(dist)[k:]] = False

        
        
        rbf = np.exp(-(z*dist)**2)
        weitghs = rbf[mask]/sum(rbf[mask]+0.0000000001)


        predicted = np.sum(weitghs*train_y[mask])

        if math.isnan(predicted):
            predicted = np.mean(train_y)


        return predicted
