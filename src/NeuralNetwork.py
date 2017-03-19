import numpy
import scipy.special
import matplotlib.pyplot


class BackPropNN:

    # initialise the neural net
    def __init__(self, inputnodes, hiddennodes, outputnodes, learningrate):
        """ Set the number of nodes in the input, hidden, and output layers """

        # Layers
        self.INPUT_NODES = inputnodes
        self.HIDDEN_NODES = hiddennodes
        self.OUTPUT_NODES = outputnodes

        # Learning rate
        self.LEARNING_RATE = learningrate

        # Link weight
        self.weight_in_hid = numpy.random.normal(0.0,pow(self.HIDDEN_NODES,-0.5),(self.HIDDEN_NODES, self.INPUT_NODES))
        self.weight_hid_out = numpy.random.normal(0.0,pow(self.OUTPUT_NODES,-0.5),(self.OUTPUT_NODES, self.HIDDEN_NODES))

        # Output error when trained
        self.correct_target = 0.0
        self.error_sum = 0.0

        # Activation Function: Sigmoid Function
        self.activation_function = lambda x: scipy.special.expit(x)

        pass

    def train(self, inputs_list, targets_list):
        """ Train the Neural Network"""

        # Convert inputs list and target list into 2d array
        inputs = numpy.array(inputs_list, ndmin=2).T
        targets = numpy.array(targets_list, ndmin=2).T

        # Calculate signals into hidden layer
        # Formula: H_inputs = W_in_hid . I
        hidden_inputs = numpy.dot(self.weight_in_hid, inputs)
        # Activate the signals in the hidden layer
        hidden_ouputs = self.activation_function(hidden_inputs)

        # Calculate signals into output layer
        # Formula: Outputs = W_hid_out . H_ouputs
        final_inputs = numpy.dot(self.weight_hid_out, hidden_ouputs)
        # Activate the signals in the output layer
        final_outputs = self.activation_function(final_inputs)

        ###### Backpropagating the Error ########

        # Output Layer Error: (target - output)
        output_errors = targets - final_outputs
        err_idx = numpy.argmax(targets)
        if err_idx == numpy.argmax(final_outputs):
            self.correct_target += 1
        self.error_sum += 1

        # Hidden: Layer Errors
        hidden_errors = numpy.dot(self.weight_hid_out.T, output_errors)

        # Update the Link Weights
        # Weights from hidden to output layer
        self.weight_hid_out += self.LEARNING_RATE * numpy.dot((output_errors * final_outputs * (1.0 - final_outputs)),
                                                              numpy.transpose(hidden_ouputs))

        # Weights from input to hidden layer
        self.weight_in_hid += self.LEARNING_RATE * numpy.dot((hidden_errors * hidden_ouputs * (1.0 - hidden_ouputs)),
                                                             numpy.transpose(inputs))

        pass

    def query(self, inputs_list):
        """ Query the Neural Network """

        # Convert input list into 2d array
        inputs = numpy.array(inputs_list, ndmin=2).T

        # Calculate signals into hidden layer
        # Formula: H_inputs = W_in_hid . I
        hidden_inputs = numpy.dot(self.weight_in_hid, inputs)
        # Activate the signals in the hidden layer
        hidden_outputs = self.activation_function(hidden_inputs)

        # Calculate signals into output layer
        # Formula: Outputs = W_hid_out . H_ouputs
        final_inputs = numpy.dot(self.weight_hid_out, hidden_outputs)
        # Activate the signals in the output layer
        final_outputs = self.activation_function(final_inputs)

        return final_outputs

    def getErrorPercent(self):
        return self.correct_target / self.error_sum

    def setWeights(self, W_i_h, W_h_o):
        self.weight_in_hid = W_i_h
        self.weight_hid_out = W_h_o


















