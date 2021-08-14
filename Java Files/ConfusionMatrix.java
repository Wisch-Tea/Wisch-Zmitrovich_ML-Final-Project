
/**
 * Responsible for managing a confusion matrix.
 */
public class ConfusionMatrix {
    
    private int[][] matrix;

    /**
     * Creates a <code>ConfusionMatrix<code/> that shows the classification
     * metrics and accuracy from a neural network. Diagonal values indicate
     * correctly classified data (assignment vs expectation/target).
     *
     * @param size
     *     Number indicating the square dimensions of the matrix to create.
     */
    public ConfusionMatrix(int size) {
        initializeMatrix(size);
    }

    /**
     * Initializes the private 2-D matrix to all zeros.
     *
     * @param size
     *     Number indicating the square dimensions of the matrix to create.
     */
    private void initializeMatrix(int size) {
        matrix = new int[size][size];
        for(int rowIndex = 0; rowIndex < size; ++rowIndex) {
            for(int columnIndex = 0; columnIndex < size; ++columnIndex) {
                matrix[rowIndex][columnIndex] = 0;
            }
        }
    }

    /**
     * Adds one to the matrix at a specific coordinate.
     *
     * @param predictedValue
     *     Number indicating matrix X coordinate.
     * @param actualValue
     *     Number indicating matrix Y coordinate.
     */
    public void add(int predictedValue, int actualValue) {
        ++matrix[predictedValue][actualValue];
    }

    /**
     * Adds one passed in confusion matrix to the current privately held matrix.
     *
     * @param toAdd
     *     ConfusionMatrix to be added to the current privately held matrix.
     */
    public void addConfusionMatrix(ConfusionMatrix toAdd) {
        for(int rowIndex = 0; rowIndex < matrix.length; ++rowIndex) {
            for(int columnIndex = 0; columnIndex < matrix.length; ++columnIndex) {
                matrix[rowIndex][columnIndex] += toAdd.matrix[rowIndex][columnIndex];
            }
        }
    }

    /**
     * Function to calculate the accuracy in the privately held class confusion matrix
     * based on the density of values along the descending diagonal cells of the matrix.
     * (Values outside of the diagonal decrease accuracy).
     *
     * @return confusion matrix accuracy
     */
    public double getAccuracy() {
        return (double)(sumPositives()) / (double)(sumPositives() + sumNegatives());
    }

    // Output the confusion matrix to terminal/cmd.
    public void print() {
        for(int rowIndex = 0; rowIndex < matrix.length; ++rowIndex) {
            for(int columnIndex = 0; columnIndex < matrix.length; ++columnIndex) {
                System.out.print("\t " + matrix[rowIndex][columnIndex]);
            }
            System.out.println();
        }
    }

    /**
     * Function to add/sum the positive classifications along the descending diagonal
     * cells of the privately held confusion matrix.
     *
     * @return numberical sum of all diagonal (positive) values.
     */
    private int sumPositives() {
        int sum = 0;
        for(int rowIndex = 0; rowIndex < matrix.length; ++rowIndex) {
            for(int columnIndex = 0; columnIndex < matrix.length; ++columnIndex) {
                sum += (rowIndex == columnIndex ? matrix[rowIndex][columnIndex] : 0);
            }
        }
        return sum;
    }

    /**
     * Function to add/sum the negative classifications that exist outside of the
     * descending diagonal cells of the privately held confusion matrix.
     *
     * @return numberical sum of all non-diagonal (negative) values.
     */
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
