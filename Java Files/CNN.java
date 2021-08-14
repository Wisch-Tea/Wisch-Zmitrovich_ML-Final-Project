import java.util.*;

/**
 * Responsible for managing a convolutional neural network.
 */
public class CNN implements Network {

    private static final int DOWNSIZE_FACTOR = 2;
    // The smaller the downsize factor, the longer it will take to execute, 
    // but also the higher the accuracy will be.
    // This value shouldn't be changed for now.

    private int inputLayerSize, hiddenLayerSize, outputLayerSize;

    private Filter[] filterSet;
    private int filterApplicationAmount;
    private MLP perceptron;

    /**
     * Creates a new <code>CNN</code> to train a model to classify an
     * American Sign Language hand sign image to an alphabetical character [A -Z].
     *
     * @param newInputLayerSize
     *     Number of input layer units that enter the neural network.
     * @param newHiddenLayerSize
     *     Number of hidden layer units that take values from the input layer and pass on a value to the output layer.
     * @param newOutputLayerSize
     *     Number of output layer units where the activation value from a hidden layer is taken to give the output classification value.
     * @param newFilterSet
     *     Filter 1-D array used in the convolution layer.
     * @param newFilterApplicationAmount
     *     Number of times each filter should be applied (further shrinking the result matrix in the process)
     */
    public CNN(int newInputLayerSize, int newHiddenLayerSize, int newOutputLayerSize, Filter[] newFilterSet, int newFilterApplicationAmount) {
        inputLayerSize = newInputLayerSize;
        hiddenLayerSize = newHiddenLayerSize;
        outputLayerSize = newOutputLayerSize;
        filterSet = newFilterSet;
        filterApplicationAmount = newFilterApplicationAmount;
        perceptron = null;
    }


    /**
     * Passes input data forward through the convolutional neural network.
     *
     * @param input
     *     Input image data object that contains pixel data and an associated label to propagate forward.
     */
    public double[] executeForwardPropagation(Input input) {
        ImageParser parser = new ImageParser();
        double[] filterSetOutput = null;
        for(int i = 0; i < filterSet.length; ++i) {
            double[] singleFilterOutput = applyFilter(filterSet[i], input, filterApplicationAmount);
            singleFilterOutput = parser.getArrayFromMatrix(parser.downsizeImage(parser.getMatrixFromArray(singleFilterOutput), DOWNSIZE_FACTOR));
            filterSetOutput = (filterSetOutput == null ? singleFilterOutput : combineTwoArrays(filterSetOutput, singleFilterOutput));
        }
        perceptron = (perceptron == null ? new MLP(filterSetOutput.length, hiddenLayerSize, outputLayerSize) : perceptron);
        return perceptron.executeForwardPropagation(new Input(filterSetOutput, input.label));
    }

    /**
     * Applies a filter to an Input data object for an 'applicationAmount' of times.
     * @param filter
     *     A Filter object containing a matrix/2-D array with [n x n] dimensions.
     * @param input
     *     Input image data object that contains pixel data and an associated label to propagate forward.
     * @param applicationAmount
     *     Number of times each filter should be applied (further shrinking the result matrix in the process)
     */
    public double[] applyFilter(Filter filter, Input input, int applicationAmount) {
        double[] filterOutput = input.data;
        for(int i = 0; i < applicationAmount; ++i) {
            filterOutput = filter.applyFilter(filterOutput);
        }
        return filterOutput;
    }

    /**
     * Passes target classification data backwards through the convolutional neural network.
     *
     * @param target
     *     1-D doubles array of target classification values to propagate backwards through the neural network.
     */
    public void executeBackPropagation(double[] target) {
        perceptron.executeBackPropagation(target);
    }

    /**
     * Sum the individual cell values of two 1-D arrays and then return the result.
     *
     * @param array1
     *     First 1-D array of values to be combined with another array.
     * @param array1
     *     Second 1-D array of values to be combined with another array.
     */
    private double[] combineTwoArrays(double[] array1, double[] array2) {
        double[] combinedArray = new double[array1.length + array2.length];
        for(int i = 0; i < array1.length; ++i) {
            combinedArray[i] = array1[i];
        }
        for(int i = 0; i < array2.length; ++i) {
            combinedArray[i + array1.length] = array2[i];
        }
        return combinedArray;
    }
    
}
