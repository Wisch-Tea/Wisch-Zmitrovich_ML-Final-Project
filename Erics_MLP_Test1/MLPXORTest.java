import java.util.*;
import java.lang.Math;


public class MLPXORTest {
    private static int[][] inputs = 
        {{0,0,0}, {0,1,1}, {1,0,1}, {1,1,0}};

    private static double biasValue = -1;
    private static double learningRate = 0.1;

    private static int inputCount = 2; // Not including bias.
    private static int hiddenCount = 2; // Not including bias.
    private static int outputCount = 1;

    private static double[][] inputToHiddenWeights;
    private static double[][] hiddenToOutputWeights;


    public static void main(String[] args) {
        initializeWeights();

        ///*
        for(int epoch = 0; epoch < 1000; ++epoch) {
            for(int i = 0; i < 4; ++i) {
                runEpoch(getInputLayer(i), getTargetValue(i));
            }
        }
        //*/
        /*
        inputToHiddenWeights[0][0] = -0.5;
        inputToHiddenWeights[0][1] = -1;
        inputToHiddenWeights[1][0] = 1;
        inputToHiddenWeights[1][1] = 1;
        inputToHiddenWeights[2][0] = 1;
        inputToHiddenWeights[2][1] = 1;
        hiddenToOutputWeights[0][0] = -0.5;
        hiddenToOutputWeights[1][0] = 1;
        hiddenToOutputWeights[2][0] = -1;
        */
        ///*
        for(int epoch = 0; epoch < 10000000; ++epoch) {
            for(int i = 0; i < 4; ++i) {
                runEpoch(getInputLayer(i), getTargetValue(i));
            }
        }

        runTest(getInputLayer(0), getTargetValue(0));
        runTest(getInputLayer(1), getTargetValue(1));
        runTest(getInputLayer(2), getTargetValue(2));
        runTest(getInputLayer(3), getTargetValue(3));

        printWeights();
        //*/
        /*
        for(int epoch = 0; epoch < 100; ++epoch) {
            System.out.println("\n*****Epoch " + (epoch + 1) + "*****");
            for(int i = 0; i < 4; ++i) {
                runEpoch(getInputLayer(i), getTargetValue(i));
                runTest(getInputLayer(0), getTargetValue(0));
                runTest(getInputLayer(1), getTargetValue(1));
                runTest(getInputLayer(2), getTargetValue(2));
                runTest(getInputLayer(3), getTargetValue(3));
                System.out.println();
            }
            System.out.println();
        }
        */
        /*
        int count = 100000000;
        while(count > 1) {
            inputToHiddenWeights[0][0] = -0.5;
            inputToHiddenWeights[0][1] = -1;
            inputToHiddenWeights[1][0] = 1;
            inputToHiddenWeights[1][1] = 1;
            inputToHiddenWeights[2][0] = 1;
            inputToHiddenWeights[2][1] = 1;
            hiddenToOutputWeights[0][0] = -0.5;
            hiddenToOutputWeights[1][0] = 1;
            hiddenToOutputWeights[2][0] = -1;

            for(int epoch = 0; epoch < count; ++epoch) {
                for(int i = 0; i < 4; ++i) {
                    runEpoch(getInputLayer(i), getTargetValue(i));
                }
            }

            runTest(getInputLayer(0), getTargetValue(0));
            runTest(getInputLayer(1), getTargetValue(1));
            runTest(getInputLayer(2), getTargetValue(2));
            runTest(getInputLayer(3), getTargetValue(3));

            printWeights();

            System.out.println("================================================================\n");

            count /= 10;
        }
        */
        /*
        inputToHiddenWeights[0][0] = 4;
        inputToHiddenWeights[0][1] = 10;
        inputToHiddenWeights[1][0] = 8;
        inputToHiddenWeights[1][1] = 7;
        inputToHiddenWeights[2][0] = 8;
        inputToHiddenWeights[2][1] = 7;
        hiddenToOutputWeights[0][0] = 9;
        hiddenToOutputWeights[1][0] = 19;
        hiddenToOutputWeights[2][0] = -19;

        runTest(getInputLayer(0), getTargetValue(0));
        runTest(getInputLayer(1), getTargetValue(1));
        runTest(getInputLayer(2), getTargetValue(2));
        runTest(getInputLayer(3), getTargetValue(3));

        printWeights();
        */
    }


    private static void printWeights() {
        System.out.println("\nFrom input to hidden...");
        System.out.println("w01: " + inputToHiddenWeights[0][0]);
        System.out.println("w02: " + inputToHiddenWeights[0][1]);
        System.out.println("w11: " + inputToHiddenWeights[1][0]);
        System.out.println("w12: " + inputToHiddenWeights[1][1]);
        System.out.println("w21: " + inputToHiddenWeights[2][0]);
        System.out.println("w22: " + inputToHiddenWeights[2][1]);
        System.out.println("\nFrom hidden to output...");
        System.out.println("w01: " + hiddenToOutputWeights[0][0]);
        System.out.println("w11: " + hiddenToOutputWeights[1][0]);
        System.out.println("w21: " + hiddenToOutputWeights[2][0]);
        System.out.println();
    }


    private static void initializeWeights() {
        inputToHiddenWeights = new double[inputCount + 1][hiddenCount];
        for(int i = 0; i < inputCount + 1; ++i) {
            for(int j = 0; j < hiddenCount; ++j) {
                inputToHiddenWeights[i][j] = getRandom();
            }
        }
        hiddenToOutputWeights = new double[hiddenCount + 1][outputCount];
        for(int i = 0; i < hiddenCount + 1; ++i) {
            for(int j = 0; j < outputCount; ++j) {
                hiddenToOutputWeights[i][j] = getRandom();
            }
        }
    }


    private static double getRandom() {
        return ((new Random().nextDouble() / 10) - 0.05);
    }


    private static double[] getInputLayer(int index) {
        double[] newInputLayer = new double[inputCount + 1];
        newInputLayer[0] = biasValue;
        for(int i = 1; i < inputCount + 1; ++i) {
            newInputLayer[i] = inputs[index][i - 1];
        }
        return newInputLayer;
    }


    private static double getTargetValue(int index) {
        return inputs[index][inputCount];
    }

    
    private static void runEpoch(double[] inputLayer, double targetValue) {
        double[] hiddenLayer = forwardInputToHidden(inputLayer);
        double output = forwardHiddenToOutput(hiddenLayer);
        double outputError = getOutputError(output, targetValue);
        double[] hiddenError = getHiddenError(hiddenLayer, outputError);
        updateHiddenToOutputWeights(outputError, hiddenLayer);
        updateInputToHiddenWeights(hiddenError, inputLayer);
    }


    private static double getOutputError(double output, double target) {
        return (output - target) * output * (1 - output);
    }


    private static double[] getHiddenError(double[] hiddenLayer, double outputError) {
        double[] newHiddenError = new double[hiddenCount + 1];
        for(int i = 0; i < hiddenCount + 1; ++i) {
            newHiddenError[i] = hiddenLayer[i] * (1 - hiddenLayer[i]) * (hiddenToOutputWeights[i][0] * outputError); 
        }
        return newHiddenError;
    }


    private static void updateHiddenToOutputWeights(double outputError, double[] hiddenLayer) {
        for(int i = 0; i < hiddenToOutputWeights.length; ++i) {
            hiddenToOutputWeights[i][0] -= learningRate * outputError * hiddenLayer[i];
        }

    }


    private static void updateInputToHiddenWeights(double[] hiddenError, double[] inputLayer) {
        for(int i = 0; i < inputToHiddenWeights.length; ++i) {
            for(int j = 1; j < hiddenError.length; ++j) {
                inputToHiddenWeights[i][j - 1] -= learningRate * hiddenError[j] * inputLayer[i];
            }
        }
    }


    /*
    private static void runTest(double[] inputLayer, double expectedValue) {
        double[] hiddenLayer = forwardInputToHidden(inputLayer);
        double output = forwardHiddenToOutput(hiddenLayer);
        System.out.print("For ((" + inputLayer[1] + ", " + inputLayer[2] + "), " + expectedValue + "), ");
        System.out.print("Output: " + output + ", ");
        if(roundOutputValue(output) == expectedValue) {
            System.out.println("[PASS]");
        } else {
            System.out.println("[FAIL]");
        }
    }
    */



    private static void runTest(double[] inputLayer, double expectedValue) {
        double[] hiddenLayer = forwardInputToHidden(inputLayer);
        System.out.println("\nHidden Layer:  ");
        for(int i = 0; i < hiddenLayer.length; ++i) {
            System.out.print(hiddenLayer[i] + "  ");
        }
        double output = forwardHiddenToOutput(hiddenLayer);
        System.out.println("\nOutput:  " + output);
        System.out.print("For ((" + inputLayer[1] + ", " + inputLayer[2] + "), " + expectedValue + "), ");
        System.out.print("Output: " + output + ", ");
        if(roundOutputValue(output) == expectedValue) {
            System.out.println("[PASS]");
        } else {
            System.out.println("[FAIL]");
        }
    }


    private static double roundOutputValue(double output) {
        if(output > 0.5) {
            return 1;
        }
        return 0;
    }


    private static double[] forwardInputToHidden(double[] inputLayer) {
        double[] newHiddenLayer = new double[hiddenCount + 1];
        newHiddenLayer[0] = biasValue;
        for(int i = 1; i < hiddenCount + 1; ++i) {
            newHiddenLayer[i] = sigmoidFunction(dotProduct(inputLayer, getWeightsAtIndex(inputToHiddenWeights, i - 1)));
        }
        return newHiddenLayer;
    }


    private static double forwardHiddenToOutput(double[] hiddenLayer) {
        return sigmoidFunction(dotProduct(hiddenLayer, getWeightsAtIndex(hiddenToOutputWeights, 0)));
    }


    private static double[] getWeightsAtIndex(double[][] weightList, int index) {
        double[] toReturn = new double[weightList.length];
        for(int i = 0; i < weightList.length; ++i) {
            toReturn[i] = weightList[i][index];
        }
        return toReturn;
    }


    private static double sigmoidFunction(double x) {
        return 1 / (1 + Math.exp(-x));
    }


    private static double dotProduct(double[] vector1, double[] vector2) {
        double sum = 0;
        for(int i = 0; i < vector1.length && i < vector1.length; ++i) {
            sum += vector1[i] * vector2[i];
        }
        return sum;
    }





}