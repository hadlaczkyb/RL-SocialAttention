import torch
import torch.nn as nn
import numpy as np
from torch.nn import functional as F
from gym import spaces

from utils.config import Configurable


#################################
### Base neural network class ###
#################################
class Base(nn.Module):
    def __init__(self, activation_type="RELU", weight_init_type="XAVIER", normalize=None):
        super().__init__()
        self.set_activations(activation_type)
        self.weight_init_type = weight_init_type
        self.normalize = normalize
        self.mean = None
        self.std = None

    @staticmethod
    def set_activations(activation_type):
        if activation_type == "RELU":
            return F.relu
        elif activation_type == "TANH":
            return F.tanh
        elif activation_type == "SIGMOID":
            return F.sigmoid
        else:
            raise ValueError("Unknown activation type!")

    def init_weights(self, m):
        if hasattr(m, 'weight'):
            if self.weight_init_type == "XAVIER":
                nn.init.xavier_uniform(m.weight.data)
            elif self.weight_init_type == "KAIMING":
                nn.init.kaiming_uniform(m.weight.data)
            elif self.weight_init_type == "ZEROS":
                nn.init.constant_(m.weight.data, 0.)
            else:
                raise ValueError("Unknown weight initialization method!")
        if hasattr(m, 'bias') and m.bias is not None:
            nn.init.constant(m.bias.data, 0.)

    def set_normalization_params(self, mean, std):
        if self.normalize:
            std[std == 0.] = 1.
        self.std = std
        self.mean = mean

    def reset(self):
        self.apply(self.init_weights)

    def forward(self, *x):
        if self.normalize:
            x = (x.float() - self.mean.float()) / self.std.float()
        return NotImplementedError


##############################################################
### Multi Layer Perceptron / Fully Connected Network class ###
##############################################################
class FullyConnectedNetwork(Base, Configurable):
    def __init__(self, config):
        super().__init__()
        Configurable.__init__(self, config)
        # self.config = config

        # Array of layer dimensions
        sizes = [self.config["in"]] + self.config["layers"]
        # Set activation function
        self.activation = self.set_activations(self.config["activation"])

        # Create neural network model
        layers_list = [nn.Linear(sizes[i], sizes[i+1]) for i in range(len(sizes) - 1)]
        self.layers = nn.ModuleList(layers_list)

        # Set output size: 'predict layer'
        if self.config.get("out", None):
            self.predict = nn.Linear(sizes[-1], self.config["out"])
        if self.config["softmax"]:
            self.softmax = nn.Softmax(dim=-1)

    @classmethod
    def default_config(cls):
        return {"in": None,
                "layers": [64, 64],
                "activation": "RELU",
                "reshape": "True",
                "out": None}

    def forward(self, x):
        if self.config["reshape"]:
            # Batch of vectors is expected
            x = x.reshape(x.shape[0], -1)
        for layer in self.layers:
            x = self.activation(layer(x))
        if self.config.get("out", None):
            x = self.predict(x)
        if self.config["softmax"]:
            x = self.softmax(x)
        return x


###################################
### Convolutional Network class ###
###################################
class ConvolutionalNetwork(Base, Configurable):
    def __init__(self, config):
        super().__init__()
        Configurable.__init__(self, config)
        # self.config = config

        # Set activation function
        self.activation = self.set_activations(self.config["activation"])

        # Define convolutional layers
        self.conv1 = nn.Conv2d(self.config["in_channels"], 16, kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=2, stride=2)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=2, stride=2)

        #**********************************************************************************
        # Function for calculating 2D convolution output size
        #   - the number of Linear input connections depends on the output of Conv2d layers
        #**********************************************************************************
        def conv2d_size_out(size, kernel_size=2, stride=2):
            return (size - (kernel_size - 1) - 1) // stride + 1

        # Get convolution output sizes
        conv_width = conv2d_size_out(conv2d_size_out(conv2d_size_out(self.config["in_width"])))
        conv_height = conv2d_size_out(conv2d_size_out(conv2d_size_out(self.config["in_height"])))

        # Check if neither the output width nor the height is zero
        assert conv_width > 0 and conv_height > 0

        # Create 'head function'
        self.config["head_fcn"]["in"] = conv_width * conv_height * 64
        self.config["head_fcn"]["out"] = self.config["out"]
        self.head = create_model(self.config["head_fcn"])

    @classmethod
    def default_config(cls):
        return {"in_channels": None,
                "in_height": None,
                "in_width": None,
                "activation": "RELU",
                "head_fcn": {
                    "type": "FullyConnectedNetwork",
                    "in": None,
                    "layers": [],
                    "activation": "RELU",
                    "reshape": "True",
                    "out": None},
                "out": None}

    def forward(self, x):
        x = self.activation((self.conv1(x)))
        x = self.activation((self.conv2(x)))
        x = self.activation((self.conv3(x)))
        return self.head(x)


#################################
### Ego Attention layer class ###
#################################
class EgoAttention(Base, Configurable):
    def __init__(self, config):
        super().__init__()
        Configurable.__init__(self, config)
        # self.config = config

        # Set feature number for each head
        self.features_per_head = int(self.config["feature_size"] / self.config["heads"])

        # V values' linear projection layer
        self.value_all = nn.Linear(self.config["feature_size"], self.config["feature_size"], bias=False)
        # K keys' linear projection layer
        self.key_all = nn.Linear(self.config["feature_size"], self.config["feature_size"], bias=False)
        # Q queries' linear projection layer
        self.query_ego = nn.Linear(self.config["feature_size"], self.config["feature_size"], bias=False)
        # Attention output layer
        self.attention_combine = nn.Linear(self.config["feature_size"], self.config["feature_size"], bias=False)

    @classmethod
    def default_config(cls):
        return {"feature_size": 64,
                "heads": 4,
                "dropout_factor": 0}

    def forward(self, ego, others, mask=None):
        batch_size = others.shape[0]
        num_entities = others.shape[1] + 1
        input_all = torch.cat((ego.view(batch_size, 1, self.config["feature_size"]), others), dim=1)

        # Dimensions: batch - entity - head - feature_per_head
        # Get all K keys
        key_all = self.key_all(input_all).view(batch_size, num_entities, self.config["heads"],
                                               self.features_per_head)
        # Get all V values
        value_all = self.value_all(input_all).view(batch_size, num_entities, self.config["heads"],
                                                   self.features_per_head)
        # Get all Q queries
        query_ego = self.query_ego(ego).view(batch_size, 1, self.config["heads"], self.features_per_head)

        # Permute K, V and Q
        key_all = key_all.permute(0, 2, 1, 3)
        value_all = value_all.permute(0, 2, 1, 3)
        query_ego = query_ego.permute(0, 2, 1, 3)

        if mask is not None:
            mask = mask.view((batch_size, 1, 1, num_entities)).repeat((1, self.config["heads"], 1, 1))

        # Get V value and attention matrix
        value, attention_matrix = attention(query_ego, key_all, value_all, mask,
                                            nn.Dropout(self.config["dropout_factor"]))
        # Get layer output
        result = (self.attention_combine(value.reshape((batch_size, self.config["feature_size"]))) + ego.squeeze(1)) / 2

        return result, attention_matrix


######################################
### Social Attention Network class ###
######################################
class SocialAttentionNetwork(Base, Configurable):
    def __init__(self, config):
        super().__init__()
        Configurable.__init__(self, config)
        self.config = config

        # Set ego encoder input size
        if not self.config["embedding_layer"]["in"]:
            self.config["embedding_layer"]["in"] = self.config["in"]
        # Set vehicle_N encoder size
        if not self.config["others_embedding_layer"]["in"]:
            self.config["others_embedding_layer"]["in"] = self.config["in"]
        # Set output decoder input size
        self.config["output_layer"]["in"] = self.config["attention_layer"]["feature_size"]
        # Set output decoder output size
        self.config["output_layer"]["out"] = self.config["out"]

        # Ego encoder layer
        self.ego_embedding = create_model(self.config["embedding_layer"])
        # Vehicle_N encoder layer
        self.others_embedding = create_model(self.config["others_embedding_layer"])
        # Ego attention layer
        self.attention_layer = EgoAttention(self.config["attention_layer"])
        # Output decoder layer
        self.output_layer = create_model(self.config["output_layer"])

    @classmethod
    def default_config(cls):
        return {
            "in": None,
            "out": None,
            "presence_feature_idx": 0,
            "embedding_layer": {
                "type": "FullyConnectedNetwork",
                "layers": [128, 128, 128],
                "reshape": False
            },
            "others_embedding_layer": {
                "type": "FullyConnectedNetwork",
                "layers": [128, 128, 128],
                "reshape": False
            },
            "attention_layer": {
                "type": "EgoAttention",
                "feature_size": 128,
                "heads": 4
            },
            "output_layer": {
                "type": "FullyConnectedNetwork",
                "layers": [128, 128, 128],
                "reshape": False
            },
        }

    def forward(self, x):
        ego_embedded_att, _ = self.forward_attention(x)
        return self.output_layer(ego_embedded_att)

    def split_input(self, x, mask=None):
        # Dims: batch - entities - features
        ego = x[:, 0:1, :]
        others = x[:, 1:, :]
        if mask is None:
            mask = x[:, :, self.config["presence_feature_idx"]:self.config["presence_feature_idx"] + 1] < 0.5
        return ego, others, mask

    def forward_attention(self, x):
        ego, others, mask = self.split_input(x)
        ego, others = self.ego_embedding(ego), self.others_embedding(others)
        return self.attention_layer(ego, others, mask)

    def get_attention_matrix(self, x):
        _, attention_matrix = self.forward_attention(x)
        return attention_matrix


###############################
### Attention Network class ###
###############################
class AttentionNetwork(Base, Configurable):
    def __init__(self, config):
        super().__init__()
        Configurable.__init__(self, config)
        self.config = config

        # Set ego encoder input size
        if not self.config["embedding_layer"]["in"]:
            self.config["embedding_layer"]["in"] = self.config["in"]
        # Set output decoder input size
        self.config["output_layer"]["in"] = self.config["attention_layer"]["feature_size"]
        # Set output decoder output size
        self.config["output_layer"]["out"] = self.config["out"]

        # Ego encoder layer
        self.embedding = create_model(self.config["embedding_layer"])
        # Output decoder layer
        self.output_layer = create_model(self.config["output_layer"])

    @classmethod
    def default_config(cls):
        return {
            "in": None,
            "out": None,
            "presence_feature_idx": 0,
            "embedding_layer": {
                "type": "FullyConnectedNetwork",
                "layers": [128, 128, 128],
                "reshape": False
            },
            "attention_layer": {
                "type": "EgoAttention",
                "feature_size": 128,
                "heads": 4
            },
            "output_layer": {
                "type": "FullyConnectedNetwork",
                "layers": [128, 128, 128],
                "reshape": False
            },
        }

    def forward(self, x):
        ego, others, mask = self.split_input(x)
        ego_embedded_att, _ = self.attention_layer(self.embedding(ego), self.others_embedding(others), mask)
        return self.output_layer(ego_embedded_att)

    def split_input(self, x):
        # Dims: batch - entities - features
        ego = x[:, 0:1, :]
        others = x[:, 1:, :]
        mask = x[:, :, self.config["presence_feature_idx"]:self.config["presence_feature_idx"] + 1] < 0.5
        return ego, others, mask

    def get_attention_matrix(self, x):
        ego, others, mask = self.split_input(x)
        _, attention_matrix = self.attention_layer(self.embedding(ego), self.others_embedding(others), mask)
        return attention_matrix


#*********************************************
# Attention computation function for each head
#*********************************************
def attention(query, key, value, mask=None, dropout=None):
    # Size of an individual feature
    d_k = query.size(-1)
    # Scores = Q*K_T/sqrt(d_k)
    scores = torch.matmul(query, key.transpose(-2, -1)) / np.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask, -1e9)
    # Attention matrix = softmax(Q*K_T/sqrt(d_k))
    p_attention = F.softmax(scores, dim=-1)
    if dropout is not None:
        p_attention = dropout(p_attention)
    # Output = softmax(Q*K_T/sqrt(d_k))*V
    output = torch.matmul(p_attention, value)
    return output, p_attention


#**********************************
# Model size configuration function
#**********************************
def size_model_config(env, model_config):
    # Get observation shape from the environment
    if isinstance(env.observation_space, spaces.Box):
        obs_shape = env.observation_space.shape
    elif isinstance(env.observation_space, spaces.Tuple):
        obs_shape = env.observation_space.spaces[0].shape

    # Set convolution sizes in case of a Convolutional Network
    if model_config["type"] == "ConvolutionalNetwork":
        model_config["in_channels"] = int(obs_shape[0])
        model_config["in_height"] = int(obs_shape[1])
        model_config["in_width"] = int(obs_shape[2])
    else:
        model_config["in"] = int(np.prod(obs_shape))

    # Set output dimensions based on the environment action space
    if isinstance(env.action_space, spaces.Discrete) and "out" not in model_config:
        model_config["out"] = env.action_space.n
    elif isinstance(env.action_space, spaces.Tuple) and "out" not in model_config:
        model_config["out"] = env.action_space.spaces[0].n


#********************************
# Create model from configuration
#********************************
def create_model(configuration: dict) -> nn.Module:
    if configuration["type"] == "FullyConnectedNetwork":
        return FullyConnectedNetwork(configuration)
    elif configuration["type"] == "ConvolutionalNetwork":
        return ConvolutionalNetwork(configuration)
    elif configuration["type"] == "SocialAttentionNetwork":
        return SocialAttentionNetwork(configuration)
    else:
        raise ValueError("Unknown model type!")


#*****************************
# Get the trainable parameters
#*****************************
def trainable_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

