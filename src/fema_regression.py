import numpy as np
import math 

class Basis:
    def __init__(self) -> None:
        pass

    def shepardBasis(train_x:np.array, train_y:np.array, test_one_sample:np.array, k:int, z:int ) -> float:
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
    

class FEMaRegressor:
    """
    Class responsible to perform the regression using FEMa approach
    """
    def __init__(self):
        self.train_x = None 
        self.train_y = None
        self.num_train_samples = 0
        self.num_features = 0
        self.k = 1
        self.basis = None
        

    def __init__(self, train_x:np.array, train_y:np.array, k:int, basis=Basis.shepardBasis) -> None:
        """Constructor that receives the train_x and train_y

        Args:
            train_x (np.array): The feature of the training set
            train_y (np.array): The target of the training set
            k (int): Define the number of neighboor used to interpolate
        """
        self.train_x = train_x
        self.train_y = train_y
        self.num_train_samples = train_x.shape[0]
        self.num_features = train_x.shape[1]
        self.k = k
        self.basis = basis

    def predict(self, test_x:np.array, *args) -> np.array:
        """Returns the prediction of the test set

        Args:
            test_x (np.array): Features of the test set
            *args : List with the specific paramter of the Basis

        Returns:
            np.array: Prediction of the test set
        """
        predicted_list = [self.basis(train_x=self.train_x, train_y=self.train_y, test_one_sample=test_x[i], k=self.k, z=args[0]) for i in range(len(test_x))]

        return predicted_list


