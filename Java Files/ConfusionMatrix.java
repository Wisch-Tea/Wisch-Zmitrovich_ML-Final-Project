public class ConfusionMatrix {
    
    private int[][] matrix;


    public ConfusionMatrix(int size) {
        initializeMatrix(size);
    }


    private void initializeMatrix(int size) {
        matrix = new int[size][size];
        for(int rowIndex = 0; rowIndex < size; ++rowIndex) {
            for(int columnIndex = 0; columnIndex < size; ++columnIndex) {
                matrix[rowIndex][columnIndex] = 0;
            }
        }
    }


    public void add(int predictedValue, int actualValue) {
        ++matrix[predictedValue][actualValue];
    }


    public void addConfusionMatrix(ConfusionMatrix toAdd) {
        for(int rowIndex = 0; rowIndex < matrix.length; ++rowIndex) {
            for(int columnIndex = 0; columnIndex < matrix.length; ++columnIndex) {
                matrix[rowIndex][columnIndex] += toAdd.matrix[rowIndex][columnIndex];
            }
        }
    }


    public double getAccuracy() {
        return (double)(sumPositives()) / (double)(sumPositives() + sumNegatives());
    }


    public void print() {
        for(int rowIndex = 0; rowIndex < matrix.length; ++rowIndex) {
            for(int columnIndex = 0; columnIndex < matrix.length; ++columnIndex) {
                System.out.print("\t" + matrix[rowIndex][columnIndex]);
            }
            System.out.println();
        }
    }


    private int sumPositives() {
        int sum = 0;
        for(int rowIndex = 0; rowIndex < matrix.length; ++rowIndex) {
            for(int columnIndex = 0; columnIndex < matrix.length; ++columnIndex) {
                sum += (rowIndex == columnIndex ? matrix[rowIndex][columnIndex] : 0);
            }
        }
        return sum;
    }


    private int sumNegatives() {
        int sum = 0;
        for(int rowIndex = 0; rowIndex < matrix.length; ++rowIndex) {
            for(int columnIndex = 0; columnIndex < matrix.length; ++columnIndex) {
                sum += (rowIndex != columnIndex ? matrix[rowIndex][columnIndex] : 0);
            }
        }
        return sum;
    }
}
