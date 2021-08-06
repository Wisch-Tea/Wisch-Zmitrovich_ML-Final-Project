import java.util.*;


/**
 * A class for objects that represent a weight vector.
 * Creates a 10-space vector with random initial values.
 * It's a child of the Layer class even though weights are technically not a "layer."
 * Because each individual weight shares a lot of the same "features" as that of a layer,
 * for the sake of this program, each weight "is a" layer.
 */
public class Weights extends Layer {
    /**
     * Default constructor.
     */
    public Weights() {
        super();
    }


    /**
     * Parameterized constructor.
     * @param newLength The length that the weight vector will be set to.
     */
    public Weights(int newLength) {
        super(newLength);
        for(int i = 0; i < newLength; ++i) {
            itemArray[i] = generateRandWeightValue();
        }
    }

    
    /**
     * Generates a random value.
     * Used to get the starting values of the weight vector.
     * @return A random double between -0.05 and 0.05.
     */
    private double generateRandWeightValue() {
        return (new Random().nextDouble() / 10) - 0.05;
    }


    /**
     * Updates the weight vector to the weight vector times a scalar value.
     * @param scalar A scalar value to scale to weight vector by.
     */
    public void multiplyByScalar(double scalar) {
        for(int i = 0; i < itemArray.length; ++i) {
            itemArray[i] = scalar * itemArray[i];
        }
    }
}
