import java.util.Vector;


/**
 * A class for objects that represent a layer in a multi-layer perceptron.
 * Manages an array of double values. 
 */
public class Layer {
    protected double[] itemArray;


    /**
     * Default constructor.
     */
    public Layer() {
        itemArray = null;
    }


    /**
     * Parameterized constructor.
     * @param arraySize The new length of itemArray.
     */
    public Layer(int arraySize) {
        itemArray = new double[arraySize];
    }


    /**
     * Retrieves the double at the given index.
     * @param index The index from which to retrieve the item.
     * @return The item from the specified index.
     */
    public double getItem(int index) {
        return itemArray[index];
    }


    /**
     * Sets the item at the given index.
     * @param index The index where the item is to be set.
     * @param newValue The new item value to be set.
     */
    public void setItem(int index, double newValue) {
        itemArray[index] = newValue;
    }


    /**
     * Creates a double array of the itemArray.
     * @return A double array copy of the items.
     */
    public double[] getCopy() {
        double[] toReturn = new double[itemArray.length];
        for(int i = 0; i < itemArray.length; ++i) {
            toReturn[i] = itemArray[i];
        }
        return toReturn;
    }


    /**
     * Sets/Resets the itemArray to be equivalent to the given double array.
     * @param newValues A double array which it will copy.
     */
    public void setItemArray(double[] newValues) {
        itemArray = new double[newValues.length];
        for(int i = 0; i < newValues.length; ++i) {
            itemArray[i] = newValues[i];
        }
    }


    /**
     * Sets/Resets the itemArray to be equivalent to the given double vector.
     */
    public void setItemArray(Vector<Double> newValues) {
        itemArray = new double[newValues.size()];
        for(int i = 0; i < newValues.size(); ++i) {
            itemArray[i] = newValues.get(i);
        }
    }


    /**
     * Extracts data from an already existing Layer object.
     * @param toCopyFrom The Layer from which it will copy data from.
     */
    public void copyFrom(Layer toCopyFrom) {
        itemArray = new double[toCopyFrom.itemArray.length];
        for(int i = 0; i < toCopyFrom.itemArray.length; ++i) {
            itemArray[i] = toCopyFrom.itemArray[i];
        }
    }


    /**
     * Creates a copied version of this Layer object.
     * @return Returned a new Layer object with copied data.
     */
    public Layer getClone() {
        Layer toReturn = new Layer();
        toReturn.itemArray = this.getCopy();
        return toReturn;
    }


    /**
     * Retrieves the size of the array of doubles stored within this object.
     * @return The size of the double array.
     */
    public int getArraySize() {
        //return (itemArray == null ? 0 : itemArray.length);
        return itemArray.length;
    }
}