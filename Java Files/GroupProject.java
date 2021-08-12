import java.util.*;


public class GroupProject {

    private static final boolean TOTAL_CONFUSION_MATRIX_PRINTING_ENABLED = true;

    private static final String IMAGES_FOLDER_PATH = "archive/signs";

    private static final int INPUT_LAYER_SIZE = 32 * 32;
    private static final int HIDDEN_LAYER_SIZE = 100;
    private static final int OUTPUT_LAYER_SIZE = 26;
    private static final int EPOCH_AMOUNT = 50;

    private static Input[] trainingSet, testingSet;
    
    // Remove old .class files: rm *.class
    //            Compile with: javac GroupProject.java
    //                Run with: java GroupProject [ARGUEMNTS]
    //             [ARGUMENTS]: "MLP" for MLP execution, and "CNN" for CNN execution.

    public static void main(String[] args) {
        System.out.println("\nBuilding training and testing sets...");
        initializeTrainingAndTestingSets();
        System.out.println("\nFinished building training and testing sets.");

        for(String arg : args) {
            if(arg.toUpperCase().equals("MLP")) {
                System.out.println("\nBeginning MLP execution...");
                executeEpochs(new MLP(INPUT_LAYER_SIZE, HIDDEN_LAYER_SIZE, OUTPUT_LAYER_SIZE));
                System.out.println("\nFinished MLP execution.");
            } else if(arg.toUpperCase().equals("CNN")) {
                System.out.println("\nBeginning CNN execution...");
                executeEpochs(new CNN(INPUT_LAYER_SIZE, HIDDEN_LAYER_SIZE, OUTPUT_LAYER_SIZE, getFilterSet()));
                System.out.println("\nFinished CNN execution.");
            }
        }
    }


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


    private static Input[] randomizeInputs(Input[] inputs) {
        List<Input> inputList = Arrays.asList(inputs);
        Collections.shuffle(inputList);
        return inputList.toArray(new Input[inputList.size()]);
    }


    private static Filter[] getFilterSet() {
        Vector<double[][]> filterMatrices = new Vector<>(4);
        filterMatrices.add(new double[][] {{0,0,0}, 
                                           {1,1,1}, 
                                           {0,0,0}});
        filterMatrices.add(new double[][] {{0,1,0}, 
                                           {0,1,0}, 
                                           {0,1,0}});
        filterMatrices.add(new double[][] {{1,0,0}, 
                                           {0,1,0}, 
                                           {0,0,1}});
        filterMatrices.add(new double[][] {{0,0,1}, 
                                           {0,1,0}, 
                                           {1,0,0}});
        Filter[] filterSet = new Filter[filterMatrices.size()];
        for(int i = 0; i < filterMatrices.size(); ++i) {
            filterSet[i] = new Filter(filterMatrices.get(i));
        }
        return filterSet;
    }


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
            System.out.print(trainingMatrix.getAccuracy() + "\t\t\t ");
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
        System.out.println();
    }


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


    private static double[] getSignVector(int signValue) {
        double[] signVector = new double[OUTPUT_LAYER_SIZE];
        for(int i = 0; i < OUTPUT_LAYER_SIZE; ++i) {
            signVector[i] = 0;
        }
        signVector[signValue] = 1;
        return signVector;
    }


    private static double dotProduct(double[] vector1, double[] vector2) {
        double sum = 0;
        for(int i = 0; i < vector1.length && i < vector1.length; ++i) {
            sum += vector1[i] * vector2[i];
        }
        return sum;
    }
}
