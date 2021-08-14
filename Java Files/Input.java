
/**
 * Responsible for managing the contents of a single input.
 * Inputs are passed into both the MLP and CNN during forward propagation.
 */
public class Input {
    
    public double[] data;
    public int label;


    public Input() {}

    /**
     * Creates a new <code>Input</code> object that takes a 1-D array of doubles
     * and an integer label to classify the data ranging: [0 - 26].
     *
     * @param newData
     *     Double 1-D array representing pixel image data to be saved.
     * @param newLabel
     *     Number indicating the assigned classification label of the data.
     */
    public Input(double[] newData, int newLabel) {
        data = newData;
        label = newLabel;
    }

    // Return the 1-D array's length value.
    public int getSize() {
        return data.length;
    }
}
