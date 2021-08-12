public interface Network {

    public double[] executeForwardPropagation(Input input);

    public void executeBackPropagation(double[] target);
}
