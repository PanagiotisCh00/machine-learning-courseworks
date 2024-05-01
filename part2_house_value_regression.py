import torch
import pickle
import numpy as np
import pandas as pd
from numpy.random import default_rng
import seaborn as sns
from sklearn.preprocessing import LabelBinarizer
from sklearn.preprocessing import StandardScaler
import torch.nn as nn
import copy

from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt

# needed for the type bug
torch.set_default_dtype(torch.float64)


class Net(nn.Module):
    """ 
        This class represents the Neural Networks. Its constructor gets as input the input size of the first layer, a list with the neurons in each of the hidden layers
        and a list with the activation functions for each of those layer. Adds in the last layer only 1 neuron with linear activation function as we predict 1 value
        - predict house prices.  
        Then it creates the network as a sequential model which is the net_stack element of the object.

        """
    def __init__(self, input_size, list_layers, activation_functions):
        super().__init__()

        layers = []
        previous_layer = input_size

        # Get starting layer and hidden layers
        for (i, (num_neurons, activation_fn)) in enumerate(zip(list_layers, activation_functions)):
          layers.append(nn.Linear(previous_layer, num_neurons))
          
          if activation_fn == "sigmoid":
            layers.append(nn.Sigmoid())
          elif activation_fn == "tanh":
            layers.append(nn.Tanh())
          elif activation_fn == "relu":
            layers.append(nn.ReLU())
          previous_layer = num_neurons

        # Get output layer, linear as it predicts a continuous value 
        layers.append(nn.Linear(previous_layer, 1))

        self.net_stack = nn.Sequential(*layers)

    def forward(self, x):
        """ 
        Forward a given input in the neural network
          
        Arguments:        
            - x {torch.tensor} -- Input of the Neural network
        Returns:    
        - logits -- Predicted result

        """
        logits = self.net_stack(x)
        return logits

class Regressor():

    def __init__(self, x, nb_epoch = 1000,learning_rate=0.01, batch_size=10,list_layers=[20, 40], activation_functions=["sigmoid", "tanh"]):
        # You can add any input parameters you need
        # Remember to set them with a default value for LabTS tests
        """ 
        Initialise the model.
          
        Arguments:
            - x {pd.DataFrame} -- Raw input data of shape 
                (batch_size, input_size), used to compute the size 
                of the network.
            - nb_epoch {int} -- number of epochs to train the network.

        """
        #######################################################################
        #                       ** START OF YOUR CODE **
        #######################################################################

        # Replace this code with your own
        X, _ = self._preprocessor(x, training = True)
        self.input_size = X.shape[1]
        self.output_size = 1
        self.nb_epoch = nb_epoch 
        
        # creating my own neural network model in the regressor 
        self.model = Net(self.input_size, list_layers, activation_functions)
        # add the learning rate and batch size
        self.lr=learning_rate
        self.batch_size=batch_size
        return

        #######################################################################
        #                       ** END OF YOUR CODE **
        #######################################################################

    def _preprocessor(self, x, y = None, training = False):
        """ 
        Preprocess input of the network.
          
        Arguments:
            - x {pd.DataFrame} -- Raw input array of shape 
                (batch_size, input_size).
            - y {pd.DataFrame} -- Raw target array of shape (batch_size, 1).
            - training {boolean} -- Boolean indicating if we are training or 
                testing the model.

        Returns:
            - {torch.tensor} or {numpy.ndarray} -- Preprocessed input array of
              size (batch_size, input_size). The input_size does not have to be the same as the input_size for x above.
            - {torch.tensor} or {numpy.ndarray} -- Preprocessed target array of
              size (batch_size, 1).
            
        """

        #######################################################################
        #                       ** START OF YOUR CODE **
        #######################################################################

        # # Replace this code with your own
        # # Return preprocessed x and y, return None for y if it was None
        # return x, (y if isinstance(y, pd.DataFrame) else None)
        # if isinstance(x,torch.tensor) :
        #      return x, y

        data_x = x.copy(deep=True)
        data_y=y
        if not isinstance(data_x, pd.DataFrame):
            return data_x,data_y     
        
        if data_y is not None:
            data_y= torch.tensor(data_y.values) 
        
        # Extract the column names
        # i use the normalization to all columns except the ocean_proximity as it will be hotfix and have values from 0 to 1 already
        columns_except_ocean_proximity = data_x.columns.difference(['ocean_proximity'])

        # Fit the scaler on the training data
        if training == True:
          self.median = data_x["total_bedrooms"].median()
          self.lb = LabelBinarizer()
          self.lb.fit(data_x['ocean_proximity'])           
          # Initialize the StandardScaler that normalized the data based only on the train input
          self.scaler = StandardScaler()
          self.scaler.fit(data_x[columns_except_ocean_proximity])
                        
        new_cols_names=self.lb.classes_ 
        new_cols_data=pd.DataFrame(self.lb.transform(data_x['ocean_proximity']), columns=new_cols_names)
        
        data_x[columns_except_ocean_proximity] = self.scaler.transform(data_x[columns_except_ocean_proximity])
        # print("cols ",new_cols_data)
        # new_cols_data=self.lb.transform(data_x['ocean_proximity'])
        # print("cNow cols are ",new_cols_data)

        data_x.drop(['ocean_proximity'], axis=1,inplace=True)
        data_x["total_bedrooms"] = data_x["total_bedrooms"].fillna(self.median)
        
        result_array = np.hstack((data_x.values, new_cols_data))

        return (torch.tensor(result_array), data_y)

        #######################################################################
        #                       ** END OF YOUR CODE **
        #######################################################################

    def train_loop(self,X,Y, loss_fn, optimizer):
        """ 
        It trains the model. It iterates the different batches of the data, makes predictions, calculate the loss, 
        does the backpropagation step to find all the intermediate differentiations, and update the weights of the Neural network 
          
        Arguments:
            - X {torch.tensor} -- preprocessed input data 
            - Y {torch.tensor} -- preprocessed results data
            - loss_fn {nn.MSELoss} -- Its the loss function which will be used in back propagation
            - optimizer {torch.optim.Adam} -- The optimizer which will be used to update the parameters of the network
            
        """
        # Set the model to training mode - important for batch normalization and dropout layers
        # Unnecessary in this situation but added for best practices
        self.model.train()
        for i in range(0, len(X), self.batch_size):
            # Compute prediction and loss
            x_batch=X[i:i+self.batch_size]
            y_batch = Y[i: i+self.batch_size]
            
            pred = self.model(x_batch)
            loss = loss_fn(pred, y_batch)
            loss.backward() 
            optimizer.step()
            optimizer.zero_grad()  
    
    def fit(self, x, y):
        """
        Regressor training function

        Arguments:
            - x {pd.DataFrame} -- Raw input array of shape 
                (batch_size, input_size).
            - y {pd.DataFrame} -- Raw output array of shape (batch_size, 1).

        Returns:
            self {Regressor} -- Trained model.

        """

        #######################################################################
        #                       ** START OF YOUR CODE **
        #######################################################################

        X, Y = self._preprocessor(x, y = y, training = True) # Do not forget
        
        loss_fn = nn.MSELoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        for epoch in range(self.nb_epoch):
            self.train_loop(X,Y, loss_fn, optimizer)
            
        return self

        #######################################################################
        #                       ** END OF YOUR CODE **
        #######################################################################

            
    def predict(self, x):
        """
        Output the value corresponding to an input x.

        Arguments:
            x {pd.DataFrame} -- Raw input array of shape 
                (batch_size, input_size).

        Returns:
            {np.ndarray} -- Predicted value for the given input (batch_size, 1).

        """

        #######################################################################
        #                       ** START OF YOUR CODE **
        #######################################################################

        X, _ = self._preprocessor(x, training = False) # Do not forget
        self.model.eval()  # Set the model to evaluation mode
        with torch.no_grad():
            y_pred = self.model(X)        
        
        return y_pred.numpy()
    
        #######################################################################
        #                       ** END OF YOUR CODE **
        #######################################################################

    def score(self, x, y):
        """
        Function to evaluate the model accuracy on a validation dataset.
        We also print more metrics for a more general evaluation process

        Arguments:
            - x {pd.DataFrame} -- Raw input array of shape 
                (batch_size, input_size).
            - y {pd.DataFrame} -- Raw output array of shape (batch_size, 1).

        Returns:
            {float} -- Quantification of the efficiency of the model.

        """

        #######################################################################
        #                       ** START OF YOUR CODE **
        #######################################################################
        X, Y = self._preprocessor(x, y, training = False) # Do not forget
        # return 0 # Replace this code with your own    
        
        y_predicted=torch.from_numpy(self.predict(x))
        
        rmse = mean_squared_error(Y, y_predicted, squared=False)
        mae = mean_absolute_error(Y, y_predicted)
        r2 = r2_score(Y, y_predicted)
        
        print("rmse ",rmse, " mae ", mae, " r2 ",r2)
        
        return mean_squared_error(Y, y_predicted)  
          
        #######################################################################
        #                       ** END OF YOUR CODE **
        #######################################################################


def save_regressor(trained_model): 
    """ 
    Utility function to save the trained regressor model in part2_model.pickle.
    """
    # If you alter this, make sure it works in tandem with load_regressor
    with open('part2_model.pickle', 'wb') as target:
        pickle.dump(trained_model, target)
    print("\nSaved model in part2_model.pickle\n")


def load_regressor(): 
    """ 
    Utility function to load the trained regressor model in part2_model.pickle.
    """
    # If you alter this, make sure it works in tandem with save_regressor
    with open('part2_model.pickle', 'rb') as target:
        trained_model = pickle.load(target)
    print("\nLoaded model in part2_model.pickle\n")
    return trained_model


def architectureHyperParameterSearch(x_train, y_train, x_val, y_val):
    """
        Function to try different architectures(neurons and layers) for the Neural Network

        Arguments:
            - x_train {torch.tensor} -- input for training data
            - y_train {torch.tensor} -- output for training data
            - x_val {torch.tensor}  -- input for validation data
            - y_val {torch.tensor}  -- output for validation data

    """
    layers = [[50],[400],[600],[10,10],[50,50],[70,70],[100,100],[150,80],[180,60],[60,180],[80,60,50],[80,100,200],[500,400,300]]
    activation_1=["relu"]
    activation_2=["sigmoid","relu"]
    activation_3=["simgoid","relu","sigmoid"]
    print("Architecture hyperparameter search")
    nb_epoch = 1500
    batch_size = 100
    learning_rate = 0.01
    best_mse=100000000
    index=0
    best_architecture=[]
    mse_results = {}  # Dictionary to store MSE for each architecture
    for layer in layers:
        if len(layer) == 1:
            activation = activation_1
        elif len(layer) == 2:
            activation = activation_2
        else:
            activation = activation_3

        regressor = Regressor(x_train, nb_epoch, learning_rate, batch_size, layer, activation)
        regressor.fit(x_train, y_train)

        # Calculate and store MSE for train and test
        print("print architecture",layer)
        print("train results:")
        mse_train = regressor.score(x_train, y_train)
        print("MSE: {}".format(mse_train))
        print("test results:")
        mse_test = regressor.score(x_val, y_val)
        print("MSE: {}".format(mse_test))
        if mse_test<best_mse or index==0:
            best_mse=mse_test
            best_architecture=layer
        index+=1
        # Store results in the dictionary
        layer_str = " ".join(map(str, layer))  # Convert layer configuration to a string
        mse_results.setdefault(layer_str, []).append(mse_test)
    
    # Plotting
    layer_labels = list(mse_results.keys())
    mse_values = [mse_results[layer][0] for layer in layer_labels]
    
    plt.bar(layer_labels,mse_values, color='blue') 
    x_positions = range(len(layer_labels))  # Generate x positions for each bar
    plt.bar(x_positions, mse_values, color='blue')
    plt.ylabel("MSE")
    plt.xlabel("layers")
    plt.xticks(x_positions, layer_labels, rotation=45)  # Set the x-ticks to be the activation labels
    plt.tight_layout()  # Adjust the layout to prevent cutting off labels
    plt.title("MSE for Different layers"+ (" ".join(map(str, layers))))
    
    plt.show()
    print("Best MSE",best_mse)
    print("Best architecture",best_architecture)



def ActivationHyperParameterSearch(x_train, y_train, x_val, y_val):
    """
    Function to try different activation functions for the 3 best Neural Network of the previous phase

    Arguments:
        - x_train {torch.tensor} -- input for training data
        - y_train {torch.tensor} -- output for training data
        - x_val {torch.tensor}  -- input for validation data
        - y_val {torch.tensor}  -- output for validation data

    """
    
    all_layers = [[50,50], [180,60], [60,80]]
    activation_all=[["relu","relu"] ,["sigmoid","sigmoid"],["relu","sigmoid"],["sigmoid","tanh"], ["relu","tanh"],["sigmoid","relu"], ["tanh","tanh"]]
    nb_epoch = 1500
    print("Activation hyperparameter search")

    batch_size = 100
    learning_rate = 0.01
    
    for layers in all_layers:

        mse_results = {}  # Dictionary to store MSE for each architecture
        best_mse=100000000
        index=0
        best_activation=[]
        for activation in activation_all:
            regressor = Regressor(x_train, nb_epoch, learning_rate, batch_size, layers, activation)
            regressor.fit(x_train, y_train)
        
            # Calculate and store MSE for train and test
            print("Activation",activation)
            print("train results:")
            mse_train = regressor.score(x_train, y_train)
            print("MSE: {}".format(mse_train))
            print("test results:")
            mse_test = regressor.score(x_val, y_val)
            print("MSE: {}".format(mse_test))
            if mse_test<best_mse or index==0:
                best_mse=mse_test
                best_activation=activation
            index+=1
            # Store results in the dictionary
            activation_str = " ".join(map(str, activation))  # Convert layer configuration to a string
            mse_results.setdefault(activation_str, []).append(mse_test)

        # Plotting
        activation_labels = list(mse_results.keys())
        
        mse_values = [mse_results[activation][0] for activation in activation_labels]
       
        plt.bar(activation_labels,mse_values, color='blue')
        x_positions = range(len(activation_labels))  # Generate x positions for each bar
        plt.bar(x_positions, mse_values, color='blue')
        plt.ylabel("MSE")
        plt.xlabel("Activations")
        plt.xticks(x_positions, activation_labels, rotation=45)  # Set the x-ticks to be the activation labels
        plt.tight_layout()  # Adjust the layout to prevent cutting off labels
        plt.title("MSE for Different Activations"+ (" ".join(map(str, layers))))
        
        # plt.show()
        plt.savefig("Graph:"+" ".join(map(str, layers)))
                    
        print("Best MSE",best_mse)
        print("Best activation",best_activation)
  
  
def LearningRateHyperParameterSearch(x_train, y_train, x_val, y_val):
    """
    Function to try different learning rates for the 3 best Neural Network of the previous phase

    Arguments:
        - x_train {torch.tensor} -- input for training data
        - y_train {torch.tensor} -- output for training data
        - x_val {torch.tensor}  -- input for validation data
        - y_val {torch.tensor}  -- output for validation data

    """

    all_layers = [[50,50], [180,60], [60,80]]
    all_activation=[["sigmoid","relu"] ,["relu","relu"],["sigmoid","relu"]]
       
    nb_epoch = 1500
    print("learning rate hyperparameter search")

    batch_size = 100
    learning_rate_all = [0.001,0.1,0.005, 0.0001,0.01]
    
    for k in range(0,len(all_layers)):
        layers=all_layers[k]
        activation=all_activation[k]
        best_mse=100000000
        index=0
        best_lr=-1
        #mse_results = {}  # Dictionary to store MSE for each architecture
        mse_results=[]
        for learning_rate in learning_rate_all:
            regressor = Regressor(x_train, nb_epoch, learning_rate, batch_size, layers, activation)
            regressor.fit(x_train, y_train)
            # Calculate and store MSE for train and test
            print("learning rate",learning_rate)
            print("train results:")
            mse_train = regressor.score(x_train, y_train)
            print("MSE: {}".format(mse_train))
            print("test results:")
            mse_test = regressor.score(x_val, y_val)
            print("MSE: {}".format(mse_test))
            if mse_test<best_mse or index==0:
                best_mse=mse_test
                best_lr=learning_rate
            index+=1
            # Store results in the dictionary
            # lr_str = " ".join(map(str, learning_rate))  # Convert layer configuration to a string
            # mse_results.setdefault(lr_str, []).append(mse_test)
            mse_results.append(mse_test)
       
        # Plotting
        # lr_labels = list(mse_results.keys())
        lr_labels=learning_rate_all
        mse_values = mse_results
        x_positions = range(len(lr_labels))  # Generate x positions for each bar
        plt.bar(x_positions, mse_values, color='blue')

        plt.ylabel("MSE")
        plt.xlabel("learning rate")
        plt.xticks(x_positions, lr_labels, rotation=45)  # Set the x-ticks to be the activation labels
        plt.tight_layout()  # Adjust the layout to prevent cutting off labels

        plt.title("MSE for Different learning rates")
        plt.savefig("Graph:"+" ".join(map(str, layers)) +" ".join(map(str, activation)))

        print("Best MSE",best_mse)
        print("Best LR",best_lr)
  
  
def BatchSizeHyperParameterSearch(x_train, y_train, x_val, y_val):
    """
    Function to try different batch sizes for the 3 best Neural Network of the previous phase

    Arguments:
        - x_train {torch.tensor} -- input for training data
        - y_train {torch.tensor} -- output for training data
        - x_val {torch.tensor}  -- input for validation data
        - y_val {torch.tensor}  -- output for validation data

    """
    
    all_layers = [[50,50], [180,60], [60,80]]
    all_activation=[["sigmoid","relu"] ,["relu","relu"],["sigmoid","relu"]]
    all_learning_rates=[0.01, 0.005, 0.005]
    nb_epoch = 1500
    print("batch size hyperparameter search")

    batch_size_all = [40, 100,1000,10000]
    
    for k in range(0,len(all_layers)):
        layers=all_layers[k]
        activation=all_activation[k]
        learning_rate = all_learning_rates[k]
        best_mse=100000000
        index=0
        best_batch=-1
        mse_results = []
    
        for batch_size in batch_size_all:
            regressor = Regressor(x_train, nb_epoch, learning_rate, batch_size, layers, activation)
            regressor.fit(x_train, y_train)
            # Calculate and store MSE for train and test
            print("Batch size",batch_size)
            print("train results:")
            mse_train = regressor.score(x_train, y_train)
            print("MSE: {}".format(mse_train))
            print("test results:")
            mse_test = regressor.score(x_val, y_val)
            print("MSE: {}".format(mse_test))
            if mse_test<best_mse or index==0:
                best_mse=mse_test
                best_batch=batch_size
            index+=1
            mse_results.append(mse_test)

        # Plotting
        batch_labels=batch_size_all
        mse_values = mse_results
        
        x_positions = range(len(batch_labels))
        
        plt.bar(x_positions, mse_values, color='blue')
        plt.ylabel("MSE")
        plt.xlabel("Batch size")
        plt.xticks(x_positions, batch_labels, rotation=45)  # Set the x-ticks to be the activation labels
        plt.tight_layout()  # Adjust the layout to prevent cutting off labels

        plt.title("MSE for Different Batch Sizes")
        plt.savefig("Graph_batch:"+" ".join(map(str, layers)) +" ".join(map(str, activation)))
        
        print("Best MSE",best_mse)
        print("Best Batch size",best_batch)
   
   
def kFoldBestExecution(x_train, y_train, x_val, y_val):
    """
    Function to train and test the 3 best Neural Network from all the different hyperparameter tuning. We used K-fold validation to make sure 
    about the correctness of the results and that they are not being overffited and printed the averages errors for each of the best networks.

    Arguments:
        - x_train {torch.tensor} -- input for training data
        - y_train {torch.tensor} -- output for training data
        - x_val {torch.tensor}  -- input for validation data
        - y_val {torch.tensor}  -- output for validation data

    """
    
    all_layers = [[50,50], [180,60], [60,80]]
    all_activation=[["sigmoid","relu"] ,["relu","relu"],["sigmoid","relu"]]
    all_learning_rates=[0.01, 0.005, 0.005]
    nb_epoch = 1500
    print("Test best Neural networks with k-fold")

    batch_size = 100
    
    random_generator=default_rng()
    n_folds = 10
    all_data_x=pd.concat([x_train, x_val], ignore_index=True)
    all_data_y=pd.concat([y_train, y_val], ignore_index=True)
    shuffled_indices = random_generator.permutation(len(all_data_x))
    split_indices = np.array_split(shuffled_indices, n_folds)
    
    for k in range(0,len(all_layers)):
        layers=all_layers[k]
        activation=all_activation[k]
        learning_rate = all_learning_rates[k]
        avg_error_train=0
        avg_error_test=0
        print("\nLayer: ",layers, "Activation:",activation, "Learning rate:",learning_rate)
        
        for n in range(n_folds):
            test_indices = split_indices[n]
            train_indices = np.hstack(split_indices[:n] + split_indices[n + 1:])
            x_train_k_fold = all_data_x.iloc[train_indices]
            y_train_k_fold = all_data_y.iloc[train_indices]
            x_test_k_fold = all_data_x.iloc[test_indices]
            y_test_k_fold = all_data_y.iloc[test_indices]
            regressor = Regressor(x_train_k_fold, nb_epoch, learning_rate, batch_size, layers, activation)
            regressor.fit(x_train_k_fold, y_train_k_fold)
            print("Training results:")
            mse_train=regressor.score(x_train_k_fold, y_train_k_fold)
            print("mse:",mse_train)
            avg_error_train += mse_train
            print("testing results")
            mse_test=regressor.score(x_test_k_fold, y_test_k_fold)
            print("mse:",mse_test)
            avg_error_test += mse_test
            
        avg_error_train=avg_error_train/n_folds
        avg_error_test=avg_error_test/n_folds
        print("average error Train:",avg_error_train)
        print("average error Test:",avg_error_test)
    
def TestManyEpochs(x_train, y_train, x_val, y_val):
    """
    Test different epochs for the best neural network.

    Arguments:
        - x_train {torch.tensor} -- input for training data
        - y_train {torch.tensor} -- output for training data
        - x_val {torch.tensor}  -- input for validation data
        - y_val {torch.tensor}  -- output for validation data

    """

    layers = [60,80]
    activation=["sigmoid","relu"]
    learning_rate=0.005
    max_epochs = 1500
    print("Epoch hyperparameter search")

    batch_size=100
    for nb_epoch in range(100,max_epochs, 100):
        regressor = Regressor(x_train, nb_epoch, learning_rate, batch_size, layers, activation)
        regressor.fit(x_train, y_train)
        print("\nEphchs:", nb_epoch)
        print("Training results:")
        mse_train=regressor.score(x_train, y_train)
        print("mse:",mse_train)
        print("testing results")
        mse_test=regressor.score(x_val, y_val)
        print("mse:",mse_test)

        
def RegressorHyperParameterSearch(x_train, y_train, x_val, y_val, x_test, y_test): 
    
    # Ensure to add whatever inputs you deem necessary to this function
    """
    Performs a hyper-parameter for fine-tuning the regressor implemented 
    in the Regressor class.

    Arguments:
        Add whatever inputs you need.
        
    Returns:
        The function should return your optimized hyper-parameters. 

    """

    #######################################################################
    #                       ** START OF YOUR CODE **
    #######################################################################
    
    # ALl steps 1-6 take a lot of time to run as it is the whole hyperparameter tuning we did!
    
    # step 1 architecture
    architectureHyperParameterSearch(x_train, y_train, x_val, y_val)
    
    # step 2 activation function
    ActivationHyperParameterSearch(x_train, y_train, x_val, y_val)
    
    # step 3 learning rate
    LearningRateHyperParameterSearch(x_train, y_train, x_val, y_val)
  
    # step 4 batch size
    BatchSizeHyperParameterSearch(x_train, y_train, x_val, y_val)
    
    # step 5: K-fold evaluation
    kFoldBestExecution(x_train, y_train, x_val, y_val)

    #step 6:epoch evaluation
    TestManyEpochs(x_train,y_train,x_val,y_val)
    
    # Best model execution and evaluation on new data:
    # [60,80] [sigmoid, relu]	learning rate=0.005, batch size=100, epochs =1100


    print("Making best model:")
    layers = [60,80]
    activation=["sigmoid","relu"]
    learning_rate=0.005
    
    nb_epoch = 1100
    batch_size = 100
    
    regressor = Regressor(x_train, nb_epoch, learning_rate, batch_size, layers, activation)
    regressor.fit(x_train, y_train)
    
    print("Training results:")
    mse_train=regressor.score(x_train, y_train)
    print("mse:",mse_train)
    
    print("Validation results:")
    mse_val=regressor.score(x_val, y_val)
    print("mse:",mse_val)
    
    print("Testing results")
    mse_test=regressor.score(x_test, y_test)
    print("mse:",mse_test)
    
    save_regressor(regressor)
        
    return  ([60,80],["sigmoid","relu"],1100,100,0.005) # Return the chosen hyper parameters

    #######################################################################
    #                       ** END OF YOUR CODE **
    #######################################################################



def example_hyper():
    output_label = "median_house_value"

    # Use pandas to read CSV data as it contains various object types
    # Feel free to use another CSV reader tool
    # But remember that LabTS tests take Pandas DataFrame as inputs
    data = pd.read_csv("housing.csv") 
      
    # Split into training, test, validation test - randomly
    # 80% training, 10% validation, 10% test
    
    percentage = 0.8
    df_train = data.sample(frac=percentage, random_state=42)  # Adjust random_state for reproducibility

    # Select the remaining 20% of rows
    df_rest = data.drop(df_train.index)  
    
    x_train = df_train.loc[:, df_train.columns != output_label]
    y_train = df_train.loc[:, [output_label]]
    
    length=df_rest.shape[0]
    half_len=int(length/2)
    
    print("length ",length," half len ",half_len)
    val=df_rest.iloc[0:half_len, :]
    
    test=df_rest.iloc[half_len:, :]
    x_validation = val.loc[:, val.columns != output_label]
    y_validation = val.loc[:, [output_label]]
    
    x_test = test.loc[:, test.columns != output_label]
    y_test = test.loc[:, [output_label]]

    
    # print("x train",x_train,"y train ",y_train)
    # print("x val ",x_validation,"y_val",y_validation)
    # print("x test",x_test," y test",y_test)
    
    # print("x train",x_train.shape,"y train ",y_train.shape)
    # print("x val ",x_validation.shape,"y_val",y_validation.shape)
    # print("x test",x_test.shape," y test",y_test.shape)

    layers,activations,epochs, batch, lr=RegressorHyperParameterSearch(x_train, y_train, x_validation, y_validation,x_test,y_test)
    

def example_main():
    output_label = "median_house_value"

    # Use pandas to read CSV data as it contains various object types
    # Feel free to use another CSV reader tool
    # But remember that LabTS tests take Pandas DataFrame as inputs
    data = pd.read_csv("housing.csv") 

    # Splitting input and output
    # x = data.loc[:, data.columns != output_label]
    # y = data.loc[:, [output_label]]

    # Training
    # This example trains on the whole available dataset. 
    # You probably want to separate some held-out data 
    # to make sure the model isn't overfitting
    
    # Split into training and test - randomly
    # 80% training, 20% testing
    
    percentage = 0.8
    df_train = data.sample(frac=percentage, random_state=42)  # Adjust random_state for reproducibility

    # Select the remaining 20% of rows
    df_test = data.drop(df_train.index)
    
    x_train = df_train.loc[:, df_train.columns != output_label]
    y_train = df_train.loc[:, [output_label]]
    
    x_test = df_test.loc[:, df_test.columns != output_label]
    y_test = df_test.loc[:, [output_label]]
    layers = [70, 50]
    activation = ["sigmoid", "relu"]
    
    regressor = Regressor(x_train, nb_epoch = 50,learning_rate=0.01, batch_size=100, list_layers=layers, activation_functions=activation)
    error_initial = regressor.score(x_train, y_train)
    
    print("\nInitial Train Regressor error : {}\n".format(error_initial))
    
    error_initial_test = regressor.score(x_test, y_test)
    
    print("\nInitial Test Regressor error: {}\n".format(error_initial_test))
    
    regressor.fit(x_train, y_train)
    save_regressor(regressor)

    # Error
    error = regressor.score(x_train, y_train)
    print("\nFinal Test Regressor error: {}\n".format(error))
    
    error_test = regressor.score(x_test, y_test)
    print("\Final Test Regressor error : {}\n".format(error_test))

if __name__ == "__main__":
    example_main()
    example_hyper()
