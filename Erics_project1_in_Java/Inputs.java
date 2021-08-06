import java.util.*;


/**
 * A class for objects that represent a single input to a perceptron.
 * It's a child of the Layer class since the input is a layer of a multi-layer perceptron.
 */
public class Inputs extends Layer {
    /**
     * The label of the input.
     * Represents the expected value that the perceptron should compute.
     */
    private int label;


    /**
     * Default constructor.
     */
    public Inputs() {
        super();
        label = 0;
    }


    /**
     * Parameterized conmstructor.
     * @param newInputArray A double array from which it will copy values from.
     * @param newLabel The new label for this input.
     */
    public Inputs(double[] newInputArray, int newLabel) {
        itemArray = new double[newInputArray.length + 1]; // + 1 for the bias.
        itemArray[0] = 1; // Adding the bias.
        for(int i = 0; i < newInputArray.length; ++i) {
            itemArray[i] = newInputArray[i];
        }
        label = newLabel;
    }


    /**
     * Parameterized conmstructor.
     * @param newInputArray A vector from which it will copy values from.
     * @param newLabel The new label for this input.
     */
    public Inputs(Vector<Double> newInputArray, int newLabel) {
        itemArray = new double[newInputArray.size() + 1]; // + 1 for the bias.
        itemArray[0] = 1; // Adding the bias.
        for(int i = 0; i < newInputArray.size(); ++i) {
            itemArray[i] = newInputArray.get(i);
        }
        label = newLabel;
    }


    /**
     * Retrieves the label of this input.
     * @return The integer value of the item's label.
     */
    public int getLabel() {
        return label;
    }


    /**
     * Creates a new Inputs object with copied data.
     * @return A new Inputs object with copied data.
     */
    public Inputs getClone() {
        Inputs toReturn = new Inputs();
        toReturn.itemArray = this.getCopy();
        toReturn.label = this.label;
        return toReturn;
    }
}