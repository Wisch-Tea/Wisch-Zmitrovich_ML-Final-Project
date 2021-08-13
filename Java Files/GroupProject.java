import java.util.*;


public class GroupProject {

    private static final boolean TOTAL_CONFUSION_MATRIX_PRINTING_ENABLED = true;

    private static final String IMAGES_FOLDER_PATH = "archive/signs";

    private static final int INPUT_LAYER_SIZE = 32 * 32;
    private static final int HIDDEN_LAYER_SIZE = 100;
    private static final int OUTPUT_LAYER_SIZE = 26;
    private static final int EPOCH_AMOUNT = 25;
    private static final int FILTER_APPLICATION_AMOUNT = 1;

    private static Input[] trainingSet, testingSet;
    
    // Remove old .class files: rm *.class
    //            Compile with: javac GroupProject.java
    //                Run with: java GroupProject [ARGUEMNTS]
    //             [ARGUMENTS]: "MLP" for MLP execution, and "CNN" for CNN execution.

    public static void main(String[] args) {
        if(args == null || args.length == 0) {
            System.out.println("No arguments provided.");
            return;
        }

        System.out.println("\nRun info:");
        System.out.println("    Amount of epochs: " + EPOCH_AMOUNT);
        System.out.println("   Hidden layer size: " + HIDDEN_LAYER_SIZE);
        System.out.println("   Amount of filters: " + getFilterSet().length);
        System.out.println("  Application amount: " + FILTER_APPLICATION_AMOUNT);

        System.out.println("\nBuilding training and testing sets...");
        initializeTrainingAndTestingSets();
        System.out.println("\nFinished building training and testing sets.");

        for(String arg : args) {
            if(arg.toUpperCase().equals("MLP")) {
                System.out.println("\nBeginning MLP execution...");
                executeEpochs(new MLP(INPUT_LAYER_SIZE, HIDDEN_LAYER_SIZE, OUTPUT_LAYER_SIZE));
                System.out.println("\nFinished MLP execution.\n");
            } else if(arg.toUpperCase().equals("CNN")) {
                System.out.println("\nBeginning CNN execution...");
                executeEpochs(new CNN(INPUT_LAYER_SIZE, HIDDEN_LAYER_SIZE, OUTPUT_LAYER_SIZE, getFilterSet(), FILTER_APPLICATION_AMOUNT));
                System.out.println("\nFinished CNN execution.\n");
            }
        }
    }

    /**
     * Generates a set of filters for a CNN.
     * @return A list of filters.
     */
    private static Filter[] getFilterSet() {
        Vector<double[][]> filterMatrices = new Vector<>();

        // Comment-out which filters are to be used or unused:

        // Top edge:
        filterMatrices.add(new double[][] {{-1,-1,-1}, 
                                           { 1, 1, 1}, 
                                           { 0, 0, 0}});
        // Bottom edge:
        filterMatrices.add(new double[][] {{ 0, 0, 0}, 
                                           { 1, 1, 1}, 
                                           {-1,-1,-1}});

        // Left side edge:
        filterMatrices.add(new double[][] {{-1, 1, 0}, 
                                           {-1, 1, 0}, 
                                           {-1, 1, 0}});
        // Right side edge:
        filterMatrices.add(new double[][] {{ 0, 1,-1}, 
                                           { 0, 1,-1}, 
                                           { 0, 1,-1}});

        // Upper-left side edge:
        filterMatrices.add(new double[][] {{-1,-1, 1}, 
                                           {-1, 1, 0}, 
                                           { 1, 0, 0}});
        // Lower-right side edge:
        filterMatrices.add(new double[][] {{ 0, 0, 1}, 
                                           { 0, 1,-1}, 
                                           { 1,-1,-1}});

        // Lower-left side edge:
        filterMatrices.add(new double[][] {{ 1, 0, 0}, 
                                           {-1, 1, 0}, 
                                           {-1,-1, 1}});
        // Upper-right side edge:
        filterMatrices.add(new double[][] {{ 1,-1,-1}, 
                                           { 0, 1,-1}, 
                                           { 0, 0, 1}});

        Filter[] filterSet = new Filter[filterMatrices.size()];
        for(int i = 0; i < filterMatrices.size(); ++i) {
            filterSet[i] = new Filter(filterMatrices.get(i));
        }
        return filterSet;
    }

    /**
     * Sets up the training and testing sets.
     * Uses ImageParser to parse date from archive/signs.
     */
    private static void initializeTrainingAndTestingSets() {
        Input[] inputs = new ImageParser().parseImages(IMAGES_FOLDER_PATH);
        Vector<Input> inputList = new Vector<>(inputs.length);
        Collections.addAll(inputList, randomizeInputs(inputs));
        testingSet = new Input[(int)(inputs.length * 0.4)];
        for(int i = 0; i < (int)(inputs.length * 0.4); ++i) {
            testingSet[i] = inputList.remove(0);
        }
        trainingSet = inputList.toArray(new Input[inputList.size()]);
    }

    /**
     * Randomizes the order of a list of inputs.
     * @param inputs The list of inputs to randomize.
     * @return The randomized list of inputs.
     */
    private static Input[] randomizeInputs(Input[] inputs) {
        List<Input> inputList = Arrays.asList(inputs);
        Collections.shuffle(inputList);
        return inputList.toArray(new Input[inputList.size()]);
    }

    /**
     * Execute the epochs over a network.
     * @param network The network to run through the epochs.
     */
    private static void executeEpochs(Network network) {
        ConfusionMatrix totalTrainingMatrix = new ConfusionMatrix(OUTPUT_LAYER_SIZE);
        ConfusionMatrix totalTestingMatrix = new ConfusionMatrix(OUTPUT_LAYER_SIZE);
        System.out.println("\nRunning Epochs...");
        System.out.println("\nOutput:");
        System.out.println("|\tEpoch\t\t|\tTraining Accuracy\t|\tTesting Accuracy\t|");
        System.out.println("\t  0\t\t 0\t\t\t\t " + executeTestingEpoch(network).getAccuracy());
        for(int i = 0; i < EPOCH_AMOUNT; ++i) {
            ConfusionMatrix trainingMatrix = executeTrainingEpoch(network);
            ConfusionMatrix testingMatrix = executeTestingEpoch(network);
            System.out.print("\t  " + (i + 1) + "\t\t ");
            System.out.print(trainingMatrix.getAccuracy() + "\t\t ");
            System.out.println(testingMatrix.getAccuracy());
            totalTrainingMatrix.addConfusionMatrix(trainingMatrix);
            totalTestingMatrix.addConfusionMatrix(testingMatrix);
        }
        System.out.println("\nFinished running Epochs.");

        if(TOTAL_CONFUSION_MATRIX_PRINTING_ENABLED == true) {
            System.out.println("\nTotal Training Confusion Matrix:\n");
            totalTrainingMatrix.print();
            System.out.println("\nTotal Testing Confusion Matrix:\n");
            totalTestingMatrix.print();
        }
    }

    /**
     * Executes a single training epoch over a netowrk.
     * @param network The network to run for a single epoch.
     * @return The confusion matrix produced over this epoch.
     */
    private static ConfusionMatrix executeTrainingEpoch(Network network) {
        ConfusionMatrix matrix = new ConfusionMatrix(OUTPUT_LAYER_SIZE);
        for(Input input : trainingSet) {
            double[] result = network.executeForwardPropagation(input);
            int predictedLabel = getPredictedLabel(result);
            int actualLabel = input.label;
            matrix.add(predictedLabel, actualLabel);
            network.executeBackPropagation(getSignVector(actualLabel));
        }
        return matrix;
    } 

    /**
     * Executes a single testing epoch over a netowrk.
     * @param network The network to run for a single epoch.
     * @return The confusion matrix produced over this epoch.
     */
    private static ConfusionMatrix executeTestingEpoch(Network network) {
        ConfusionMatrix matrix = new ConfusionMatrix(OUTPUT_LAYER_SIZE);
        for(Input input : testingSet) {
            double[] result = network.executeForwardPropagation(input);
            int predictedLabel = getPredictedLabel(result);
            int actualLabel = input.label;
            matrix.add(predictedLabel, actualLabel);
        }
        return matrix;
    } 

    /**
     * Finds the sign label whose respective label is "closest" to a predicted vector.
     * It generates a label for a given prediction vector.
     * @param preditedVector The prediction vector.
     * @return The label assigned for the prediction vector.
     */
    private static int getPredictedLabel(double[] preditedVector) {
        int predictedLabel = 0;
        double predictedDotProduct = 0;
        for(int currentSign = 0; currentSign < OUTPUT_LAYER_SIZE; ++currentSign) {
            double dotProduct = dotProduct(preditedVector, getSignVector(currentSign));
            if(dotProduct > predictedDotProduct) {
                predictedLabel = currentSign;
                predictedDotProduct = dotProduct;
            }
        }
        return predictedLabel;
    }

    /**
     * Generates a vector in 26-space corresponding to a given label.
     * @param signLabel The label of the desired vector.
     * @return The vector corresponding to the label.
     */
    private static double[] getSignVector(int signLabel) {
        double[] signVector = new double[OUTPUT_LAYER_SIZE];
        for(int i = 0; i < OUTPUT_LAYER_SIZE; ++i) {
            signVector[i] = 0;
        }
        signVector[signLabel] = 1;
        return signVector;
    }

    /**
     * Performs a dot product over 2 "vectors".
     * @param vector1 The first vector.
     * @param vector2 The second vector.
     * @return The dot product result.
     */
    private static double dotProduct(double[] vector1, double[] vector2) {
        double sum = 0;
        for(int i = 0; i < vector1.length && i < vector1.length; ++i) {
            sum += vector1[i] * vector2[i];
        }
        return sum;
    }
}
