import java.util.*;


public class CNN {

    private static final int INPUT_LAYER_SIZE = 32 * 32;
    private static final int HIDDEN_LAYER_SIZE = 100;
    private static final int OUTPUT_LAYER_SIZE = 26;

    private Filter[] filterSet;
    private MLP[] perceptronSet;


    public CNN() {
        constructFilterSet();
        constructPerceptronSet();
    }


    private void constructFilterSet() {
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
        filterSet = new Filter[filterMatrices.size()];
        for(int i = 0; i < filterMatrices.size(); ++i) {
            filterSet[i] = new Filter(filterMatrices.get(i));
        }
    }


    private void constructPerceptronSet() {
        perceptronSet = new MLP[filterSet.length];
        for(int i = 0; i < filterSet.length; ++i) {
            perceptronSet[i] = new MLP(INPUT_LAYER_SIZE, HIDDEN_LAYER_SIZE, OUTPUT_LAYER_SIZE);
        }
    }


    public double[] executeForwardPropagation(Input input) {
        double[][] outputSet = new double[filterSet.length][OUTPUT_LAYER_SIZE];
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
