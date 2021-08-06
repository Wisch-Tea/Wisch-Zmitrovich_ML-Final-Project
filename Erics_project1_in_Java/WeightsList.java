public class WeightsList extends LayerList {
    /**
     * Default constructor.
     */
    public WeightsList() {
        layerList = null;
    }


    /**
     * Parameterized constructor.
     * @param sendingUnitCount The size of the layer of nodes that will be "sending" through the weights.
     * @param recievingUnitCount The size of the laye of nodes that will be "recieving" from the weights.
     */
    public WeightsList(int sendingUnitCount, int recievingUnitCount) {
        layerList = new Weights[sendingUnitCount];
        for(int i = 0; i < sendingUnitCount; ++i) {
            layerList[i] = new Weights(recievingUnitCount);
        }
    }


    /**
     * Creates a copied version of this WeightsList.
     * @return A WeightsList object with copied data.
     */
    public WeightsList getClone() {
        WeightsList toReturn = new WeightsList();
        toReturn.layerList = this.getCopy();
        return toReturn;
    }
}
