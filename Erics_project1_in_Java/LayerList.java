public class LayerList {
    protected Layer[] layerList;


    /**
     * Default constructor.
     */
    public LayerList() {
        layerList = null;
    }


    /**
     * Adds an item to the already full layerList.
     * @param toAdd The new Layer object to add to the list.
     */
    public void addItem(Layer toAdd) {
        if(layerList == null) {
            layerList = new Layer[1];
            layerList[0] = toAdd.getClone();
        } else {
            Layer[] newLayerList = new Layer[layerList.length + 1];
            for(int i = 0; i < layerList.length; ++i) {
                newLayerList[i] = layerList[i].getClone();
            }
            newLayerList[layerList.length] = toAdd.getClone();
            layerList = newLayerList;
        }
    }


    /**
     * Replaces the layerList with a new layerList.
     * @param newLayerList The new layerList to replace the current layerList.
     */
    public void insertLayerList(Layer[] newLayerList) {
        layerList = newLayerList.clone();
    }


    /**
     * Retrieves the length of the layerList.
     * @return The size of layerList.
     */
    public int getListLength() {
        return layerList.length;
    }


    /**
     * Retrieves a Layer object from the layerList.
     * @param index The index from which to retieve.
     * @return The Layer object from the specified index.
     */
    public Layer getItem(int index) {
        return layerList[index];
    }


    /**
     * Creates a copy of layerList.
     * @return A Layer array copy of LayerList.
     */
    public Layer[] getCopy() {
        Layer[] toReturn = new Layer[layerList.length];
        for(int i = 0; i < layerList.length; ++i) {
            toReturn[i] = layerList[i].getClone();
        }
        return toReturn;
    }


    /**
     * Creates a LayerList object with the same data as this.
     * @return A LayerList object with copied data.
     */
    public LayerList getClone() {
        LayerList toReturn = new LayerList();
        toReturn.layerList = this.getCopy();
        return toReturn;
    }
}
