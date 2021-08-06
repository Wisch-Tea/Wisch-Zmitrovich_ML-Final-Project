import java.util.*;
import java.lang.Math;


// Compile with: javac Program1.java
//               java Program1


public class Program1 {
    private static String trainImageFileName = "train-images.idx3-ubyte";
    private static String trainLabelFileName = "train-labels.idx1-ubyte";
    private static String testImageFileName = "t10k-images.idx3-ubyte";
    private static String testLabelFileName = "t10k-labels.idx1-ubyte";

    private static InputsList trainingInputs;
    private static InputsList testingInputs;

    private static int inputLayerSize = 785; // Bias included.
    private static int epochMax = 50;

    private static double learningRate = 0.1;
    private static double momentum = 0.9;

    /**
     * The class used to pass data while the MLP is running.
     */
    private static class RunData {
        public WeightsList inputToHidden; 
        public WeightsList hiddenToOutput; 
        public int numberOfHidden; 
        public ConfusionMatrix totalTrainingMatrix; 
        public ConfusionMatrix epochTrainingMatrix;
        public ConfusionMatrix totalTestingMatrix; 
        public ConfusionMatrix epochTestingMatrix;
        public MLPData previousMLPData;
        public RunData() {
            inputToHidden = null;
            hiddenToOutput = null;
            numberOfHidden = 0;
            totalTrainingMatrix = null;
            epochTrainingMatrix = null;
            totalTestingMatrix = null;
            epochTestingMatrix = null;
            previousMLPData = null;
        }
    }

    /**
     * The class used to store MLP data after it's been propagated forward. 
     */
    private static class MLPData {
        public Inputs inputLayer;
        public WeightsList inputToHidden;
        public Layer hiddenLayer;
        public WeightsList hiddenToOutput;
        public Layer outputLayer;
        public MLPData() {
            inputLayer = null;
            inputToHidden = null;
            hiddenLayer = null;
            hiddenToOutput = null;
            outputLayer = null;
        }
        public MLPData(Inputs newInputLayer, Layer newHiddenLayer, Layer newOutputLayer, RunData runData) {
            inputLayer = newInputLayer.getClone();
            inputToHidden = runData.inputToHidden.getClone();
            hiddenLayer = newHiddenLayer.getClone();
            hiddenToOutput = runData.hiddenToOutput.getClone();
            outputLayer = newOutputLayer.getClone();
        }
    }


    public static void main(String[] args) {
        System.out.println("\nBuilding training inputs...");
        trainingInputs = new InputsList(trainImageFileName, trainLabelFileName);
        System.out.println("\nFinished building training inputs.");
        System.out.println("\nBuilding testing inputs...");
        testingInputs = new InputsList(testImageFileName, testLabelFileName);
        System.out.println("\nFinished building testing inputs.");

        System.out.println("\nRunning Experiment #1...");
        experiment1();
        System.out.println("\nFinished running Experiment #1.");

        System.out.println("\nRunning Experiment #2...");
        experiment2();
        System.out.println("\nFinished running Experiment #2.");

        System.out.println();
    }


    private static void experiment1() {
        // For 20 hidden units:
        System.out.println("\nExperiment #1: Running MLP with 20 hidden units...");
        run(20);
        System.out.println("\nExperiment #1: Finished running MLP with 20 hidden units.");
        // For 50 hidden units:
        System.out.println("\nExperiment #1: Running MLP with 50 hidden units...");
        run(50);
        System.out.println("\nExperiment #1: Finished running MLP with 50 hidden units.");
        // For 100 hidden units:
        System.out.println("\nExperiment #1: Running MLP with 100 hidden units...");
        run(100);
        System.out.println("\nExperiment #1: Finished running MLP with 100 hidden units.");
    }


    private static void experiment2() {
        InputsList originalTrainingInputs;
        // For 1/4 of training examples:
        System.out.println("\nExperiment #2: Generating training examples of 1/4 of the size...");
        originalTrainingInputs = trainingInputs;
        splitTrainingInputs(4);
        System.out.println("\nExperiment #2: Finished generating training examples of 1/4 of the size.");
        System.out.println("\nExperiment #2: Running MLP with 1/4 of training examples...");
        run(100);
        System.out.println("\nExperiment #2: Finished running MLP with 1/4 of training examples.");
        trainingInputs = originalTrainingInputs;
        // For 1/2 of training examples:
        System.out.println("\nExperiment #2: Generating training examples of 1/2 of the size...");
        originalTrainingInputs = trainingInputs;
        splitTrainingInputs(2);
        System.out.println("\nExperiment #2: Finished generating training examples of 1/2 of the size.");
        System.out.println("\nExperiment #2: Running MLP with 1/2 of training examples...");
        run(100);
        System.out.println("\nExperiment #2: Finished running MLP with 1/2 of training examples.");
        trainingInputs = originalTrainingInputs;
    }


    /**
     * Splits the training set for Experiment #2.
     * @param divisor The factor by which to split the training set.
     */
    private static void splitTrainingInputs(int divisor) {
        int originalSize = trainingInputs.getListLength();
        int newSize = (int)(originalSize / divisor);
        Vector<Integer> indexList = new Vector<>(newSize);
        Layer[] layerList = new Layer[newSize];

        int[] labelList = new int[] {0,0,0,0,0,0,0,0,0,0};   
        int indexCount = 0;
        while(indexCount < newSize) {
            int randomIndex = new Random().nextInt(originalSize);
            int newLabel = trainingInputs.getItem(randomIndex).getLabel();
            if(labelList[newLabel] < (int)(newSize / 10) && indexList.contains(randomIndex) == false) {
                layerList[indexCount] = trainingInputs.getItem(randomIndex).getClone();
                indexList.add(randomIndex);
                ++labelList[newLabel];
                ++indexCount;
            }
        }
        trainingInputs.insertLayerList(layerList);
    }


    /**
     * "Runs" the MLP.
     * @param numberOfHiddenUnits The number of hidden units the MLP will have for this run.
     */
    private static void run(int numberOfHiddenUnits) {
        // The intput layer has 784+1 units, 
        // so there will be a total of 785 weight groupings from the input to hidden layer.
        // The number of hidden units will vary between 20, 50, and 100,
        // so each input unit will have 20, 50, or 100 weights going from it to the hidden layer units.
        // Going from the hidden layer to the output layer, 
        // there will be either 20+1, 50+1, or 100+1 weight groupings going from the hidden layer to the output.
        // Each hidden unit will have 10 weights going from it to each unit in the output.
        RunData runData = new RunData(); // Create a runData object to store the data needed for each run.
        runData.inputToHidden = new WeightsList(inputLayerSize, numberOfHiddenUnits);
        runData.hiddenToOutput = new WeightsList(numberOfHiddenUnits + 1, 10); // +1 for bias.
        runData.numberOfHidden = numberOfHiddenUnits;
        runData.totalTrainingMatrix = new ConfusionMatrix();
        runData.totalTestingMatrix = new ConfusionMatrix();

        System.out.println("\nTraining...");
        System.out.println("\nOutput:");
        System.out.println("|\tEpoch\t\t|\tTraining Accuracy\t|\tTesting Accuracy\t|");
        runEpochs(runData);
        System.out.println("\nFinished training.");
        System.out.println("\nTraining Confusion Matrix:");
        runData.totalTrainingMatrix.print();
        System.out.println("\nTesting Confusion Matrix:");
        runData.totalTestingMatrix.print();
    }


    /**
     * Runs goes through the epoch count and calls functions to compute training and testing per each epoch.
     * @param runData The run data that is needed for each run.
     */
    public static void runEpochs(RunData runData) {
        for(int i = 0; i < epochMax; ++i) {
            runData.epochTrainingMatrix = new ConfusionMatrix();
            runData.epochTestingMatrix = new ConfusionMatrix();
            if(i == 0) { // Make sure to run testing input before training begins.
                testingRun(runData);
                System.out.println("\t  0\t\t 0\t\t\t\t " + runData.epochTestingMatrix.getAccuracy());
                runData.epochTestingMatrix = new ConfusionMatrix();
            }
            trainingRun(runData);
            testingRun(runData);
            System.out.print("\t  " + (i+1) + "\t\t ");
            System.out.print(runData.epochTrainingMatrix.getAccuracy() + "\t\t\t ");
            System.out.println(runData.epochTestingMatrix.getAccuracy());
        }
    }


    /**
     * A training run. Runs through the traing inputs list and calls trainingStep().
     * @param runData The The run data that is needed for each run.
     */
    private static void trainingRun(RunData runData) {
        for(int i = 0; i < trainingInputs.getListLength(); ++i) {
            trainingStep((Inputs)trainingInputs.getItem(i), runData);
        }
    }


    /**
     * A single "step" in the training run. 
     * Calls functions relating to the forward and back-propagation steps.
     * @param inputLayer The current input that will be passed through the MLP for this step.
     * @param runData The run data that is needed for each run.
     */
    private static void trainingStep(Inputs inputLayer, RunData runData) {
        // Forward-propagation:
        Layer hiddenLayer = forwardInputToHidden(inputLayer, runData.inputToHidden, runData.numberOfHidden);
        Layer outputLayer = forwardHiddenToOutput(hiddenLayer, runData.hiddenToOutput, runData.numberOfHidden);
        int expectedValue = inputLayer.getLabel();
        int actualValue = findActualValue(outputLayer.getCopy());
        runData.totalTrainingMatrix.add(expectedValue, actualValue);
        runData.epochTrainingMatrix.add(expectedValue, actualValue);
        if(expectedValue != actualValue) { // i.e. incorrect guess...
            // Back-propagation:
            MLPData mlpData = new MLPData(inputLayer, hiddenLayer, outputLayer, runData);
            updateWeights(expectedValue, mlpData, runData);
            runData.previousMLPData = mlpData;
        }
    }


    /**
     * A testing run. Runs through the testing inputs list and calls trainingStep().
     * @param runData The run data that is needed for each run.
     */
    private static void testingRun(RunData runData) {
        for(int i = 0; i < testingInputs.getListLength(); ++i) {
            testingStep((Inputs)testingInputs.getItem(i), runData);
        }
    }


    /**
     * A single "step" in the testing run. 
     * Calls functions relating to the forward propagation step.
     * @param inputLayer The current input that will be passed through the MLP for this step.
     * @param runData The run data that is needed for each run.
     */
    private static void testingStep(Inputs inputLayer, RunData runData) {
        // Forward propagation:
        Layer hiddenLayer = forwardInputToHidden(inputLayer, runData.inputToHidden, runData.numberOfHidden);
        Layer outputLayer = forwardHiddenToOutput(hiddenLayer, runData.hiddenToOutput, runData.numberOfHidden);
        int expectedValue = inputLayer.getLabel();
        int actualValue = findActualValue(outputLayer.getCopy());
        runData.totalTestingMatrix.add(expectedValue, actualValue);
        runData.epochTestingMatrix.add(expectedValue, actualValue);
    }


    //------------------- Forward-Propagation Section -------------------
    //================================================================
    //vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv
    /**
     * The forward propagation step between the input layer and the hidden layer.
     * @param inputLayer The current inputLayer.
     * @param inputToHidden The weights between the input layer and the hidden layer.
     * @param numberOfHidden The number of hidden units in the hidden layer.
     * @return The calculated hidden layer.
     */
    private static Layer forwardInputToHidden(Inputs inputLayer, WeightsList inputToHidden, int numberOfHidden) {
        Layer hiddenLayer = new Layer(numberOfHidden + 1); // + 1 for the bias.
        hiddenLayer.setItem(0, 1); // Set the bias.
        // Calculating each hidden unit by adding together the products of each input value
        // and its weight that's connected to the current hidden unit and passing it through the sigmoid function.
        for(int hiddenIndex = 0; hiddenIndex < numberOfHidden; ++hiddenIndex) {
            double hiddenUnitSum = 0;
            for(int inputIndex = 0; inputIndex < inputLayerSize; ++inputIndex) {
                hiddenUnitSum += inputLayer.getItem(inputIndex) * inputToHidden.getItem(inputIndex).getItem(hiddenIndex);
            }
            hiddenLayer.setItem(hiddenIndex + 1, sigmoidFunction(hiddenUnitSum));
        }
        return hiddenLayer;
    }


    /**
     * The forward propagation step between the hidden layer and the output layer.
     * @param hiddenLayer The hidden layer (calculated from the forward propagation step between the input layer and the hidden layer).
     * @param hiddenToOutput The weights between the hidden layer and the output layer.
     * @param numberOfHidden The number of hidden units in the hidden layer.
     * @return The calculated output layer.
     */
    private static Layer forwardHiddenToOutput(Layer hiddenLayer, WeightsList hiddenToOutput, int numberOfHidden) {
        Layer outputLayer = new Layer(10);
        // Calculating each output unit by adding together the products of each hidden value
        // and its weight that's connected to the current output unit and passing it through the sigmoid function.
        for(int outputIndex = 0; outputIndex < 10; ++outputIndex) {
            double outputSum = 0;
            for(int hiddenIndex = 0; hiddenIndex < numberOfHidden + 1; ++hiddenIndex) {
                outputSum += hiddenLayer.getItem(hiddenIndex) * hiddenToOutput.getItem(hiddenIndex).getItem(outputIndex);
            }
            outputLayer.setItem(outputIndex, sigmoidFunction(outputSum));
        }
        return outputLayer;
    }
    //^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    //================================================================
    //----------------------------------------------------------------


    //------------------- Back-Propagation Section -------------------
    //================================================================
    //vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv
    /**
     * Updates both hidden-to-output and input-to-hidden weights.
     * @param expectedValue The value that the output was expected to evaluate to.
     * @param mlpData The data of the current MLP.
     * @param runData The run data that is needed for each run.
     */
    private static void updateWeights(int expectedValue, MLPData mlpData, RunData runData) {
        double[] outputErrorTerms = getOutputErrorTerms(expectedValue, mlpData);
        double[] hiddenErrorTerms = getHiddenErrorTerms(outputErrorTerms, mlpData);
        updateHiddenToOutputWeights(outputErrorTerms, mlpData, runData);
        updateInputToHiddenWeights(hiddenErrorTerms, mlpData, runData);
    }


    /**
     * Calculates the error terms of the output layer (error_k = output_k(1 - output_k)(target_k - output_k)).
     * @param expectedValue The value that the output was expected to evaluate to.
     * @param mlpData The data of the current MLP.
     * @return The list of the output layer error terms.
     */
    private static double[] getOutputErrorTerms(int expectedValue, MLPData mlpData) {
        double[] toReturn = new double[10]; // 10 output units, so 10 error terms.
        double[] targetVector = getTargetVector(expectedValue);
        for(int i = 0; i < 10; ++i) {
            double outputUnit = mlpData.outputLayer.getItem(i);
            double targetUnit = targetVector[i];
            toReturn[i] = outputUnit * (1 - outputUnit) * (targetUnit - outputUnit);
        }
        return toReturn;
    }


    /**
     * Calculates the error terms of the hidden layer (error_j = hidden_j(1 - hidden_j)(dot-product(hidden-to-output-weights_j, error_of_output))).
     * @param outputErrorTerms The output layer error terms (calculated in getOutputErrorTerms()).
     * @param mlpData The data of the current MLP.
     * @return The list of the hidden layer error terms.
     */
    private static double[] getHiddenErrorTerms(double[] outputErrorTerms, MLPData mlpData) {
        double[] toReturn = new double[mlpData.hiddenLayer.getArraySize() - 1]; // -1 for bias.
        for(int i = 1; i < mlpData.hiddenLayer.getArraySize() - 1; ++i) { // Need to start at i=1 to skip bias.
            double hiddenUnit = mlpData.hiddenLayer.getItem(i);
            double[] weightsVector = mlpData.hiddenToOutput.getItem(i).getCopy();
            toReturn[i] = hiddenUnit * (1 - hiddenUnit) * dotProduct(outputErrorTerms, weightsVector); 
        }
        return toReturn;
    }


    /**
     * Updates the hidden-to-output weights (delta_weight_j = learning-rate * output-error-terms * hidden-unit + momentum * delta_weight_j^t-1).
     * @param outputErrorTerms The output layer error terms (calculated in getOutputErrorTerms()).
     * @param mlpData The data of the current MLP.
     * @param runData The run data that is needed for each run.
     */
    private static void updateHiddenToOutputWeights(double[] outputErrorTerms, MLPData mlpData, RunData runData) {
        for(int hiddenLayerIndex = 0; hiddenLayerIndex < mlpData.hiddenLayer.getArraySize(); ++hiddenLayerIndex) {
            for(int weightIndex = 0; weightIndex < 10; ++weightIndex) {
                double currentWeightValue = runData.hiddenToOutput.getItem(hiddenLayerIndex).getItem(weightIndex);
                double newWeightValue = currentWeightValue + (learningRate * outputErrorTerms[weightIndex] * mlpData.hiddenLayer.getItem(hiddenLayerIndex));
                if(runData.previousMLPData != null) { // Only add momentum if the previous MLP exists.
                    newWeightValue += momentum * (currentWeightValue - runData.previousMLPData.hiddenToOutput.getItem(hiddenLayerIndex).getItem(weightIndex));
                }
                runData.hiddenToOutput.getItem(hiddenLayerIndex).setItem(weightIndex, newWeightValue);
            }
        }
    }


    /**
     * Updates the input-to-hidden weights (delta_weight_j = learning-rate * hidden-error-terms * input-unit + momentum * delta_weight_j^t-1).
     * @param hiddenErrorTerms The hidden layer error terms (calculated in getHiddenErrorTerms()).
     * @param mlpData The data of the current MLP.
     * @param runData The run data that is needed for each run.
     */
    private static void updateInputToHiddenWeights(double[] hiddenErrorTerms, MLPData mlpData, RunData runData) {
        for(int inputLayerIndex = 0; inputLayerIndex < mlpData.inputLayer.getArraySize(); ++inputLayerIndex) {
            for(int weightIndex = 0; weightIndex < mlpData.hiddenLayer.getArraySize() - 1; ++weightIndex) { // -1 for bias.
                double currentWeightValue = runData.inputToHidden.getItem(inputLayerIndex).getItem(weightIndex);
                double newWeightValue = currentWeightValue + (learningRate * hiddenErrorTerms[weightIndex] * mlpData.inputLayer.getItem(inputLayerIndex));
                if(runData.previousMLPData != null) { // Only add momentum if the previous MLP exists.
                    newWeightValue += momentum * (currentWeightValue - runData.previousMLPData.inputToHidden.getItem(inputLayerIndex).getItem(weightIndex));
                }
                runData.inputToHidden.getItem(inputLayerIndex).setItem(weightIndex, newWeightValue);
            }
        }
    }
    //^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    //================================================================
    //----------------------------------------------------------------


    /**
     * Calculates the values of an output array.
     * Does this by finding which target vector has the biggest dot-product with output vector.
     * @param outputVector The output array from which the value will be calculated.
     * @return The calculated value.
     */
    private static int findActualValue(double[] outputVector) {
        int currentMatch = 0;
        double currentMatchValue = 0;
        for(int i = 0; i < 10; ++i) {
            double matchValue = dotProduct(outputVector, getTargetVector(i));
            if(matchValue > currentMatchValue) {
                currentMatch = i;
                currentMatchValue = matchValue;
            }
        }
        return currentMatch;
    }
    

    /**
     * Gets the specified target vector.
     * @param target Target vector selection.
     * @return The selected target.
     */
    private static double[] getTargetVector(int target) {
        double[] toReturn = new double[10];
        for(int i = 0; i < 10; ++i) {
            // Every index that is not representing the value of the target vector is set to 0.1.
            toReturn[i] = 0.1;
        }
        // The index that does represent the value of the target vector is set to 0.9.
        toReturn[target] = 0.9;
        return toReturn;
    }


    /**
     * The sigmoid function (1/(1+e^-x)).
     * @param input The x value of the sigmoid function.
     * @return The result of the sigmoid function.
     */
    public static double sigmoidFunction(double input) {
        return 1 / (1 + Math.exp(-1 * input));
    }


    /**
     * Calculates the dot-product between two vectors.
     * @param vector1 The first vector.
     * @param vector2 The second vector.
     * @return The dot-product between the first vector and the second vector.
     */
    public static double dotProduct(double[] vector1, double[] vector2) {
        double sum = 0;
        for(int i = 0; i < vector1.length && i < vector2.length; ++i) {
            sum += vector1[i] * vector2[i];
        }
        return sum;
    } 
}
