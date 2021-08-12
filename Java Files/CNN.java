import java.util.*;


public class CNN implements Network {

    private int inputLayerSize, hiddenLayerSize, outputLayerSize;

    private Filter[] filterSet;
    private MLP[] perceptronSet;


    public CNN(int newInputLayerSize, int newHiddenLayerSize, int newOutputLayerSize, Filter[] newFilterSet) {
        inputLayerSize = newInputLayerSize;
        hiddenLayerSize = newHiddenLayerSize;
        outputLayerSize = newOutputLayerSize;
        filterSet = newFilterSet;
        constructPerceptronSet();
    }


    private void constructPerceptronSet() {
        perceptronSet = new MLP[filterSet.length];
        for(int i = 0; i < filterSet.length; ++i) {
            perceptronSet[i] = new MLP(inputLayerSize, hiddenLayerSize, outputLayerSize);
        }
    }


    public double[] executeForwardPropagation(Input input) {
        double[][] outputSet = new double[filterSet.length][outputLayerSize];
        for(int i = 0; i < filterSet.length; ++i) {
            Input filteredInput = new Input(filterSet[i].applyFilter(input.data), input.label);
            outputSet[i] = perceptronSet[i].executeForwardPropagation(filteredInput);
        }
        return combineOutputs(outputSet);
    }


    private double[] combineOutputs(double[][] outputs) {
        double[] toReturn = new double[outputs[0].length];
        for(int i = 0; i < outputs[0].length; ++i) {
            double sum = 0;
            for(int j = 0; j < outputs.length; ++j) {
                sum += outputs[j][i];
            }
            toReturn[i] = sum / outputs.length;
        }
        return toReturn;
    }


    public void executeBackPropagation(double[] target) {
        for(int i = 0; i < perceptronSet.length; ++i) {
            perceptronSet[i].executeBackPropagation(target);
        }
    }
    
}
