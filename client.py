# Import the necessary libraries
from centralized import load_data, load_model, train, test
from collections import OrderedDict
import flwr as fl # Import the Flower library
import torch


"""
Utility function to set the hyperparameters of the model
Since we need to update the parameters (built-in in TensorFlow)
"""
def set_parameters(model, parameters):
    # Get the dictionary of the model's state, and the keys.
    # We are zipping it, with the updated parameters
    params_dict = zip(model.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
    # We are setting the strict to True, since the model should always have the same parameters and shape.
    model.load_state_dict(state_dict, strict=True)


net = load_model() # Load the model
train_set, test_set = load_data() # Load the data

class FlowerClient(fl.client.NumPyClient):
    # We need to write this function our self, since it isn't implemented in pytorch
    def get_parameters(self, config):
        # We are returning the parameters of the model
        # We are converting them to numpy arrays, and returning them
        return [val.cpu().numpy() for _, val in net.state_dict().items()]
    
    def fit(self, parameters, config):
        # Setting the parameters, to the paramters we've received from the server
        set_parameters(net, parameters)
        train(net, train_set, epochs=1) # Train the model for 1 epoch (can be adjusted)
        # If we want to define a metric, we should define it in the TRAIN function
        return self.get_parameters({}), len(train_set.dataset), {} # Return the parameters, the number of samples, and an metric (empty for now)
    
    def evaluate(self, parameters, config):
        set_parameters(net, parameters) # Set the parameters given from the server
        loss, accuracy = test(net, test_set) # Test the model
        return float(loss), len(test_set.dataset), {"accuracy": float(accuracy)} # Return the loss, the number of samples, and the accuracy
    

fl.client.start_numpy_client(
    server_address="127.0.0.1:8080",
    client=FlowerClient(),
) # Start the client