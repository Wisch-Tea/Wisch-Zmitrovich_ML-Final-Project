import java.util.*;

/**
 * Responsible for managing a mulit-later perceptron.
 */
public class MLP implements Network {

    private final boolean MOMENTUM_ENABLED = false;

    private final double BIAS_VALUE = 1;
    private final double LEARNING_RATE = 0.1;
    private final double MOMENTUM_VALUE = 0.9;

    private int inputLayerSize, hiddenLayerSize, outputLayerSize;
    private double[][] inputToHiddenWeights, hiddenToOutputWeights;
    private double[][] deltaInputToHiddenWeights, deltaHiddenToOutputWeights;
    private double[] hiddenLayer, inputLayer, outputLayer;


    public MLP(int newInputLayerSize, int newHiddenLayerSize, int newOutputLayerSize) {
        inputLayerSize = newInputLayerSize;
        hiddenLayerSize = newHiddenLayerSize;
        outputLayerSize = newOutputLayerSize;
        inputLayer = null;
        hiddenLayer = null;
        outputLayer = null;
        initializeWeights();
    }


    private void initializeWeights() {
        // +1 for the weights from the bias nodes.
        inputToHiddenWeights = new double[inputLayerSize + 1][hiddenLayerSize];
        deltaInputToHiddenWeights = new double[inputLayerSize + 1][hiddenLayerSize];
        hiddenToOutputWeights = new double[hiddenLayerSize + 1][outputLayerSize];
        deltaHiddenToOutputWeights = new double[hiddenLayerSize + 1][outputLayerSize];
        Random random = new Random();
        for(int i = 0; i < inputLayerSize + 1; ++i) {
            for(int j = 0; j < hiddenLayerSize; ++j) {
                inputToHiddenWeights[i][j] = (random.nextDouble() / 10) - 0.05;
                deltaInputToHiddenWeights[i][j] = 0;
            }
        }
        for(int i = 0; i < hiddenLayerSize + 1; ++i) {
            for(int j = 0; j < outputLayerSize; ++j) {
                hiddenToOutputWeights[i][j] = (random.nextDouble() / 10) - 0.05;
                deltaHiddenToOutputWeights[i][j] = 0;
            }
        }
    }


    public double[] executeForwardPropagation(Input input) {
        inputLayer = input.data;
        hiddenLayer = calculateNextLayer(inputLayer, inputToHiddenWeights, hiddenLayerSize);
        outputLayer = calculateNextLayer(hiddenLayer, hiddenToOutputWeights, outputLayerSize);
        return outputLayer;
    }


    private double[] calculateNextLayer(double[] layer, double[][] weights, int nextLayerSize) {
        layer = getBiasedLayer(layer); // Add bias to layer.
        double[] nextLayer = new double[nextLayerSize];
        for(int nextLayerIndex = 0; nextLayerIndex < nextLayerSize; ++nextLayerIndex) {
            double total = 0;
            for(int layerIndex = 0; layerIndex < layer.length; ++layerIndex) {
                total += layer[layerIndex] * weights[layerIndex][nextLayerIndex];
            }
            nextLayer[nextLayerIndex] = sigmoidFunction(total);
        }
        return nextLayer;
    }


    private double[] getBiasedLayer(double[] originalLayer) {
        double[] toReturn = new double[originalLayer.length + 1]; // +1 for bias.
        toReturn[0] = BIAS_VALUE;
        for(int i = 0; i < originalLayer.length; ++i) {
            toReturn[i + 1] = originalLayer[i];
        }
        return toReturn;
    }


    private static double sigmoidFunction(double x) {
        return 1 / (1 + Math.exp(-x));
    }


    public void executeBackPropagation(double[] target) {
        double[] outputError = calculateOutputError(target);
        double[] hiddenError = calculateHiddenError(outputError);
        updateHiddenToOutputWeights(outputError);
        updateInputToHiddenWeights(hiddenError);
        inputLayer = null;
        hiddenLayer = null;
        outputLayer = null;
    }


    private double[] calculateOutputError(double[] target) {
        double[] outputError = new double[outputLayerSize];
        for(int i = 0; i < outputLayer.length; ++i) {
            outputError[i] = outputLayer[i] * (1 - outputLayer[i]) * (target[i] - outputLayer[i]);
        }
        return outputError;
    }


    private double[] calculateHiddenError(double[] outputError) {
        double[] hiddenError = new double[hiddenLayerSize];
        for(int i = 0; i < hiddenLayerSize; ++i) {
            double summation = 0;
            for(int j = 0; j < outputLayerSize; ++j) {
                summation += hiddenToOutputWeights[i + 1][j] * outputError[j];
            }
            hiddenError[i] = hiddenLayer[i] * (1 - hiddenLayer[i]) * summation;
        }
        return hiddenError;
    }


    private void updateHiddenToOutputWeights(double[] outputError) {
        double[] biasedHiddenLayer = getBiasedLayer(hiddenLayer);
        for(int i = 0; i < hiddenLayerSize + 1; ++i) {
            for(int j = 0; j < outputLayerSize; ++j) {
                double newValue = hiddenToOutputWeights[i][j] + (LEARNING_RATE * outputError[j] * biasedHiddenLayer[i]);
                if(MOMENTUM_ENABLED == true) {
                    newValue += deltaHiddenToOutputWeights[i][j] * MOMENTUM_VALUE;
                }
                deltaHiddenToOutputWeights[i][j] = newValue - hiddenToOutputWeights[i][j];
                hiddenToOutputWeights[i][j] = newValue;
            }
        }
    }


    private void updateInputToHiddenWeights(double[] hiddenError) {
        double[] biasedInputLayer = getBiasedLayer(inputLayer);
        for(int i = 0; i < inputLayerSize + 1; ++i) {
            for(int j = 0; j < hiddenLayerSize; ++j) {
                double newValue = inputToHiddenWeights[i][j] + (LEARNING_RATE * hiddenError[j] * biasedInputLayer[i]);
                if(MOMENTUM_ENABLED == true) {
                    newValue += deltaInputToHiddenWeights[i][j] * MOMENTUM_VALUE;
                }
                deltaInputToHiddenWeights[i][j] = newValue - inputToHiddenWeights[i][j];
                inputToHiddenWeights[i][j] = newValue;
            }
        }
    }
}
