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


    public CNN(int newInputLayerSize, int newHiddenLayerSize, int newOutputLayerSize, Filter[] newFilterSet, int newFilterApplicationAmount) {

        inputLayerSize = newInputLayerSize;
        hiddenLayerSize = newHiddenLayerSize;
        outputLayerSize = newOutputLayerSize;
        filterSet = newFilterSet;
        filterApplicationAmount = newFilterApplicationAmount;
        perceptron = null;
    }


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


    public double[] applyFilter(Filter filter, Input input, int applicationAmount) {
        double[] filterOutput = input.data;
        for(int i = 0; i < applicationAmount; ++i) {
            filterOutput = filter.applyFilter(filterOutput);
        }
        return filterOutput;
    }


    public void executeBackPropagation(double[] target) {
        perceptron.executeBackPropagation(target);
    }


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
