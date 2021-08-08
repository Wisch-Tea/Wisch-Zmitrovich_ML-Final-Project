public class Input {
    
    public double[] data;
    public int label;


    public Input() {}


    public Input(double[] newData, int newLabel) {
        data = newData;
        label = newLabel;
    }


    public int getSize() {
        return data.length;
    }
}
