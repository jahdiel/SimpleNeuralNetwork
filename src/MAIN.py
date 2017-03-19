
""" BackPropNN: Main Function in the Neural Network"""

import numpy

import time
import scipy.misc

####### Train the Neural Network ########

def trainNet(epochs, save=False):
    # Load the MNIST training data CSV file into a list
    train_data_file = open("D:\NeuralNetworks\MNIST_Dataset\Training_Set\mnist_train.csv", 'r')

    # Epochs is the number of times the training data set is used for training
    for e in xrange(epochs):
        print 'Epoch #'+str(e)
        # Go through all test in the training data set
        for idx, test in enumerate(train_data_file):

            # Split the test by the ',' commas
            all_values = test.split(',')

            # scale and shift the inputs
            inputs = (numpy.asfarray(all_values[1:]) / 255 * 0.99) + 0.01

            # Create the target output (all 0.01, except the desired label which is 0.99)
            targets = numpy.zeros(OUTPUT_NODES) + 0.1
            # all_values[0] is the target label for this test
            targets[int(all_values[0])] = 0.99

            # Train the neural net
            backProp.train(inputs, targets)


            pass

        print 100 - (backProp.getErrorPercent() * 100)

        backProp.correct_target = 0.0
        backProp.error_sum = 0.0

        # Reset the file
        train_data_file.seek(0)

        pass
    # Close training data set file
    train_data_file.close()

    if save:
        # Save the trained weights
        wih1 = ','.join(map(str, numpy.ravel(backProp.weight_in_hid1)))
        wh1h2 = ','.join(map(str, numpy.ravel(backProp.weight_hid1_hid2)))
        wh2o = ','.join(map(str, numpy.ravel(backProp.weight_hid2_out)))

        f = open('LinkWeights_i784_h200_h50_o10_L0-03.txt', 'w')
        # W_I_H -> (hidden, input)
        f.write(str(HIDDEN_NODES)+','+str(INPUT_NODES)+'\n')
        f.write(wih1)
        # W_I_H -> (hidden, input)
        f.write("\n"+ str(HIDDEN_NODES_2) + ',' + str(HIDDEN_NODES) + '\n')
        f.write(wh1h2)
        # W_H_O -> (output, hidden)
        f.write("\n"+str(OUTPUT_NODES)+','+str(HIDDEN_NODES_2)+'\n')
        f.write(wh2o)

##############################################

def queryNet():
    # Load the MNIST test data CSV file into a list
    test_data_file = open("D:\NeuralNetworks\MNIST_Dataset\Test_Set\mnist_test.csv", 'r')

    ########### Test the Neural Network ##########

    # Scorecard for how well the network performs
    scorecard = []
    score_correct = 0.0
    total_test = 0

    # Verify all tests in the test data set
    for test in test_data_file:
        # split the record by the ',' commas
        all_values = test.split(',')
        # correct answer is first value
        correct_label = int(all_values[0])
        # scale and shift the inputs
        inputs = (numpy.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01
        # query the network
        outputs = backProp.query(inputs)
        # the index of the highest value corresponds to the label
        label = numpy.argmax(outputs)
        # append correct or incorrect to list
        if (label == correct_label):
            # network's answer matches correct answer, add 1 to scorecard
            scorecard.append(1)
            score_correct += 1

        else:
            # network's answer doesn't match correct answer, add 0 to scorecard
            scorecard.append(0)

        total_test += 1

        pass

    # Close test data set file
    test_data_file.close()

    # calculate the performance score, the fraction of correct answers
    scorecard_array = numpy.asarray(scorecard)

    print "performance =", float(scorecard_array.sum())/scorecard_array.size

    ##############################################

def ourOwnDigits():
    """ Test our Own Handwritten Digits"""
    # Use saved weights
    LW_from_txt()

    # Draw the digit
    import DrawNumber

    # Import the digit image
    img_array = scipy.misc.imread("numberTest.png", flatten=True)
    img_data = 255.0 - img_array.reshape(784)
    img_data = (img_data / 255.0 * 0.99) + 0.01

    # query the network
    outputs = backProp.query(img_data)
    # the index of the highest value corresponds to the label
    label = numpy.argmax(outputs)

    return label, outputs


def LW_from_txt():
    # Load the link weights from txt file
    link_weights_file = open('LinkWeights_i784_h200_o10_L0-1_.txt', 'r')
    weight_list = []
    for index, weight in enumerate(link_weights_file):
        if (index % 2 == 0):
            dimensions = map(int, weight.split(','))
        else:
            w = numpy.asfarray(map(float, weight.split(','))).reshape(tuple(dimensions))
            weight_list.append(w)

    backProp.setWeights(weight_list[0], weight_list[1])



if __name__ == '__main__':

    from NeuralNetwork import *
    from BackPropNN import *

    # Number of Layer Nodes
    INPUT_NODES = 784
    HIDDEN_NODES = 200
    HIDDEN_NODES_2 = 50
    OUTPUT_NODES = 10

    # Epochs
    EPOCHS = 5
    # Learning rate
    LEARNING_RATE = 0.5

    # Momentum
    MOMENTUM = 0.3

    # Create instance of BackPropNN
    backProp = BackPropNN(INPUT_NODES, HIDDEN_NODES, OUTPUT_NODES, LEARNING_RATE)
    # backProp = BackPropNN_4L(INPUT_NODES, HIDDEN_NODES, HIDDEN_NODES_2, OUTPUT_NODES, LEARNING_RATE)


    # Test our own handwritten digits
    output, output_arr = ourOwnDigits()
    print "Output Performance:"
    print output_arr
    print "The digit is:", output
    """
    start_time = time.time()
    # Train the Neural Net
    trainNet(EPOCHS)
    # Query the Neural Net
    queryNet()
    final_time = time.time()

    print "Time taken:", final_time - start_time, "secs."
    """









