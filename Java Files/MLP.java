import java.util.*;

/**
 * Responsible for creating and managing a multi-layer perceptron.
 */
public class MLP implements Network {
    /*============================================ CONFIG CONSTANTS ============================================ */
    // Train with momentum term selection. [Enable to find the global error minimum]
    private final boolean MOMENTUM_ENABLED = true;

    private final double BIAS_VALUE = 1;
    private final double LEARNING_RATE = 0.1;
    private final double MOMENTUM_VALUE = 0.9;

    private int inputLayerSize, hiddenLayerSize, outputLayerSize;
    private double[] hiddenLayer, inputLayer, outputLayer;
    private double[][] inputToHiddenWeights, hiddenToOutputWeights;
    private double[][] deltaInputToHiddenWeights, deltaHiddenToOutputWeights;

    /**
     * Creates a new <code>MLP</code> to train a model to classify an
     * American Sign Language hand sign image to an alphabetical character [A -Z].
     *
     * @param newInputLayerSize
     *     Number of input layer units that enter the neural network.
     * @param newHiddenLayerSize
     *     Number of hidden layer units that take values from the input layer and pass on a value to the output layer.
     * @param newOutputLayerSize
     *     Number of output layer units where the activation value from a hidden layer is taken to give the output classification value.
     */
    public MLP(int newInputLayerSize, int newHiddenLayerSize, int newOutputLayerSize) {
        inputLayerSize = newInputLayerSize;
        hiddenLayerSize = newHiddenLayerSize;
        outputLayerSize = newOutputLayerSize;
        inputLayer = null;
        hiddenLayer = null;
        outputLayer = null;
        initializeWeights();
    }

    /**
     * Randomly initializes the weight values between neural network layers.
     */
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

    /**
     * Passes input data forward through the MLP.
     *
     * @param input
     *     Input image data object that contains pixel data and an associated label to propagate forward.
     */
    public double[] executeForwardPropagation(Input input) {
        inputLayer = input.data;
        hiddenLayer = calculateNextLayer(inputLayer, inputToHiddenWeights, hiddenLayerSize);
        outputLayer = calculateNextLayer(hiddenLayer, hiddenToOutputWeights, outputLayerSize);
        return outputLayer;
    }

    /**
     * Creates a new <code>CNN</code> to train a model to classify an
     * American Sign Language hand sign image to an alphabetical character [A -Z].
     *
     * @param layer
     *     1-D doubles array that represents the units within a layer.
     * @param weights
     *     2-D doubles array that represents the weight relationships between the
     *     previous layer units and the current layer units.
     * @param nextLayerSize
     *     Number indicating the amount of units in the next layer in the neural network.
     */
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

    /**
     * Adds the pre-determined bias value to the layer calculations.
     *
     * @param originalLayer
     *     1-D doubles array that represents the units within a layer.
     */
    private double[] getBiasedLayer(double[] originalLayer) {
        double[] toReturn = new double[originalLayer.length + 1]; // +1 for bias.
        toReturn[0] = BIAS_VALUE;
        for(int i = 0; i < originalLayer.length; ++i) {
            toReturn[i + 1] = originalLayer[i];
        }
        return toReturn;
    }

    /**
     * The sigmoid/activation function determining whether to output 0 or 1.
     *
     * @param x
     *     Double value used to determine activation or not.
     */
    private static double sigmoidFunction(double x) {
        return 1 / (1 + Math.exp(-x));
    }

    /**
     * Passes target classification data backwards through the MLP.
     *
     * @param target
     *     1-D doubles array of target classification values to propagate backwards through the neural network.
     */
    public void executeBackPropagation(double[] target) {
        double[] outputError = calculateOutputError(target);
        double[] hiddenError = calculateHiddenError(outputError);
        updateHiddenToOutputWeights(outputError);
        updateInputToHiddenWeights(hiddenError);
        inputLayer = null;
        hiddenLayer = null;
        outputLayer = null;
    }

    /**
     * Calculates the error from the output layer's classification labels vs the expected target labels.
     *
     * @param target
     *     1-D double array of classified input value labels to determine error.
     */
    private double[] calculateOutputError(double[] target) {
        double[] outputError = new double[outputLayerSize];
        for(int i = 0; i < outputLayer.length; ++i) {
            outputError[i] = outputLayer[i] * (1 - outputLayer[i]) * (target[i] - outputLayer[i]);
        }
        return outputError;
    }

    /**
     * Calculates the error from the hidden layer's classification labels vs the expected target labels.
     *
     * @param outputError
     *     1-D double array of output layer error calculations to determine hidden layer error.
     */
    private double[] calculateHiddenError(double[] outputError) {
        double[] hiddenError = new double[hiddenLayerSize];
        for(int i = 0; i < hiddenLayerSize; ++i) {
            double summation = 0;
            for(int j = 0; j < outputLayerSize; ++j) {
                summation += hiddenToOutputWeights[i + 1][j] * outputError[j]; // +1 for the bias.
            }
            hiddenError[i] = hiddenLayer[i] * (1 - hiddenLayer[i]) * summation;
        }
        return hiddenError;
    }

    /**
     * Updates the output layer's weights based on the passed in pre-calculated output error.
     *
     * @param outputError
     *     1-D double array of output layer error calculations to determine new output layer weights.
     */
    private void updateHiddenToOutputWeights(double[] outputError) {
        double[] biasedHiddenLayer = getBiasedLayer(hiddenLayer);
        for(int i = 0; i < hiddenLayerSize + 1; ++i) { // +1 for the bias.
            for(int j = 0; j < outputLayerSize; ++j) {
                double newValue = hiddenToOutputWeights[i][j] + (LEARNING_RATE * outputError[j] * biasedHiddenLayer[i]);
                if(MOMENTUM_ENABLED == true) {
                    newValue += deltaHiddenToOutputWeights[i][j] * MOMENTUM_VALUE;
                }
                deltaHiddenToOutputWeights[i][j] = hiddenToOutputWeights[i][j] - newValue;
                hiddenToOutputWeights[i][j] = newValue;
            }
        }
    }

    /**
     * Updates the hidden layer's weights based on the passed in pre-calculated hidden layer error.
     *
     * @param outputError
     *     1-D double array of hidden layer error calculations to determine new hidden layer weights.
     */
    private void updateInputToHiddenWeights(double[] hiddenError) {
        double[] biasedInputLayer = getBiasedLayer(inputLayer);
        for(int i = 0; i < inputLayerSize + 1; ++i) { // +1 for the bias.
            for(int j = 0; j < hiddenLayerSize; ++j) {
                double newValue = inputToHiddenWeights[i][j] + (LEARNING_RATE * hiddenError[j] * biasedInputLayer[i]);
                if(MOMENTUM_ENABLED == true) {
                    newValue += deltaInputToHiddenWeights[i][j] * MOMENTUM_VALUE;
                }
                deltaInputToHiddenWeights[i][j] = inputToHiddenWeights[i][j] - newValue;
                inputToHiddenWeights[i][j] = newValue;
            }
        }
    }
}