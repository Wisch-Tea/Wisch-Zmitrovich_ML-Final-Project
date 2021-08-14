
/**
 * The outline of the functions that a network possess.
 * MLP and CNN are children of this class, therefore, they
 * both contain functions to forward and back propagate.
 */
public interface Network {

    public double[] executeForwardPropagation(Input input);

    public void executeBackPropagation(double[] target);
}
