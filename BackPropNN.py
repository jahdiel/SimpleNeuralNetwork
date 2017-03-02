import numpy
import scipy.special
import matplotlib.pyplot


class BackPropNN_4L:
    """ This a backpropagation neural network of 4 layers"""

    # initialise the neural net
    def __init__(self, inputnodes, hiddennodes1, hiddennodes2, outputnodes, learningrate):
        """ Set the number of nodes in the input, hidden, and output layers """

        # Layers
        self.INPUT_NODES = inputnodes
        self.HIDDEN_NODES_1 = hiddennodes1
        self.HIDDEN_NODES_2 = hiddennodes2
        self.OUTPUT_NODES = outputnodes

        # Learning rate
        self.LEARNING_RATE = learningrate

        # Link weight
        self.weight_in_hid1 = numpy.random.normal(0.0, pow(self.HIDDEN_NODES_1, -0.5), (self.HIDDEN_NODES_1, self.INPUT_NODES))
        self.weight_hid1_hid2 = numpy.random.normal(0.0, pow(self.HIDDEN_NODES_2, -0.5), (self.HIDDEN_NODES_2, self.HIDDEN_NODES_1))
        self.weight_hid2_out = numpy.random.normal(0.0, pow(self.OUTPUT_NODES, -0.5), (self.OUTPUT_NODES, self.HIDDEN_NODES_2))

        # Activation Function: Sigmoid Function
        self.activation_function = lambda x: scipy.special.expit(x)

        pass

    def train(self, inputs_list, targets_list):
        """ Train the Neural Network"""

        # Convert inputs list and target list into 2d array
        inputs = numpy.array(inputs_list, ndmin=2).T
        targets = numpy.array(targets_list, ndmin=2).T

        # Calculate signals into hidden layer 1
        # Formula: H_inputs = W_in_hid . I
        hidden_inputs_1 = numpy.dot(self.weight_in_hid1, inputs)
        # Activate the signals in the hidden layer
        hidden_outputs_1 = self.activation_function(hidden_inputs_1)

        # Calculate signals into hidden layer 2
        hidden_inputs_2 = numpy.dot(self.weight_hid1_hid2, hidden_outputs_1)
        # Activate the signals in the hidden layer
        hidden_outputs_2 = self.activation_function(hidden_inputs_2)

        # Calculate signals into output layer
        # Formula: Outputs = W_hid_out . H_ouputs
        final_inputs = numpy.dot(self.weight_hid2_out, hidden_outputs_2)
        # Activate the signals in the output layer
        final_outputs = self.activation_function(final_inputs)

        ###### Backpropagating the Error ########

        # Output Layer Error: (target - output)
        output_errors = targets - final_outputs

        # Hidden1: Layer Errors
        hidden_errors_2 = numpy.dot(self.weight_hid2_out.T, output_errors)

        # Hidden1: Layer Errors
        hidden_errors_1 = numpy.dot(self.weight_hid1_hid2.T, hidden_errors_2)

        # Update the Link Weights
        # Weights from hidden to output layer
        self.weight_hid2_out += self.LEARNING_RATE * numpy.dot((output_errors * final_outputs * (1.0 - final_outputs)),
                                                               numpy.transpose(hidden_outputs_2))

        # Weights from hidden to output layer
        self.weight_hid1_hid2 += self.LEARNING_RATE * numpy.dot((hidden_errors_2 * hidden_outputs_2 * (1.0 - hidden_outputs_2)),
                                                               numpy.transpose(hidden_outputs_1))

        # Weights from input to hidden layer
        self.weight_in_hid1 += self.LEARNING_RATE * numpy.dot((hidden_errors_1 * hidden_outputs_1 * (1.0 - hidden_outputs_1)),
                                                              numpy.transpose(inputs))

        pass

    def query(self, inputs_list):
        """ Query the Neural Network """

        # Convert input list into 2d array
        inputs = numpy.array(inputs_list, ndmin=2).T

        # Calculate signals into hidden layer 1
        # Formula: H_inputs = W_in_hid . I
        hidden_inputs_1 = numpy.dot(self.weight_in_hid1, inputs)
        # Activate the signals in the hidden layer
        hidden_outputs_1 = self.activation_function(hidden_inputs_1)

        # Calculate signals into hidden layer 2
        hidden_inputs_2 = numpy.dot(self.weight_hid1_hid2, hidden_outputs_1)
        # Activate the signals in the hidden layer
        hidden_outputs_2 = self.activation_function(hidden_inputs_2)

        # Calculate signals into output layer
        # Formula: Outputs = W_hid_out . H_ouputs
        final_inputs = numpy.dot(self.weight_hid2_out, hidden_outputs_2)
        # Activate the signals in the output layer
        final_outputs = self.activation_function(final_inputs)

        return final_outputs


    def setWeights(self, W_i_h, W_h_h, W_h_o):
        self.weight_in_hid1 = W_i_h
        self.weight_hid1_hid2 = W_h_h
        self.weight_hid2_out = W_h_o


















