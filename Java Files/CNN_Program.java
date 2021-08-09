import java.util.*;


public class CNN_Program {

    private static final String IMAGES_FOLDER_PATH = "archive/signs";

    private static final int INPUT_LAYER_SIZE = 32 * 32;
    private static final int HIDDEN_LAYER_SIZE = 100;
    private static final int OUTPUT_LAYER_SIZE = 26;
    private static final int EPOCH_AMOUNT = 50;

    private static Input[] trainingSet, testingSet;

    private static CNN network;
    
    // Remove old .class files: rm *.class
    //            Compile with: javac CNN_Program.java
    //                Run with: java CNN_Program 

    public static void main(String[] args) {
        System.out.println("\nBuilding training and testing sets...");
        initializeTrainingAndTestingSets();
        System.out.println("\nFinished building training and testing sets.");

        network = new CNN();

        System.out.println("\nRunning Epochs...");
        System.out.println("\nOutput:");
        System.out.println("|\tEpoch\t\t|\tTraining Accuracy\t|\tTesting Accuracy\t|");
        System.out.println("\t  0\t\t 0\t\t\t\t " + executeTestingEpoch().getAccuracy());
        for(int i = 0; i < EPOCH_AMOUNT; ++i) {
            ConfusionMatrix trainingMatrix = executeTrainingEpoch();
            ConfusionMatrix testingMatrix = executeTestingEpoch();
            System.out.print("\t  " + (i + 1) + "\t\t ");
            System.out.print(trainingMatrix.getAccuracy() + "\t\t\t ");
            System.out.println(testingMatrix.getAccuracy());
        }
        System.out.println("\nFinished running Epochs.");
        System.out.println();
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


    private static ConfusionMatrix executeTrainingEpoch() {
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


    private static ConfusionMatrix executeTestingEpoch() {
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
