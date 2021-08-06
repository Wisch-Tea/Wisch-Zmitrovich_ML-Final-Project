public class ConfusionMatrix {
    int[][] matrix;


    /**
     * Default constructor.
     */
    public ConfusionMatrix() {
        initialize();
    }


    /**
     * Sets initilizes the matrix to be 10x10 of 0 values.
     */
    public void initialize() {
                        //      Expected Values
                        //   |---------------------|
                                                     // ---
        matrix = new int[][] {{0,0,0,0,0,0,0,0,0,0}, //  | 
                              {0,0,0,0,0,0,0,0,0,0}, //  |
                              {0,0,0,0,0,0,0,0,0,0}, //  |
                              {0,0,0,0,0,0,0,0,0,0}, //  |
                              {0,0,0,0,0,0,0,0,0,0}, //  | Actual Values
                              {0,0,0,0,0,0,0,0,0,0}, //  |
                              {0,0,0,0,0,0,0,0,0,0}, //  |
                              {0,0,0,0,0,0,0,0,0,0}, //  |
                              {0,0,0,0,0,0,0,0,0,0}, //  |
                              {0,0,0,0,0,0,0,0,0,0}};//  |
    }                                                // ---


    /**
     * Adds a new point to the confusion matrix.
     * @param expectedValue The expected value as an integer.
     * @param actualValue The actual value as an integer.
     */
    public void add(int expectedValue, int actualValue) {
        matrix[actualValue][expectedValue] += 1;
    }


    /**
     * Gives the current accuracy of the confusion matrix.
     * @return The amount of current results / Amount of items.
     */
    public double getAccuracy() {
        return (((double)sumPositives()) / (double)((sumPositives() + sumNegatives())));
    }


    /**
     * Sums the amount of "correct" items.
     * @return The count of the positives.
     */
    private int sumPositives() {
        int sum = 0;
        for(int y = 0; y < 10; ++y) {
            for(int x = 0; x < 10; ++x) {
                if(x == y) {
                    sum += matrix[y][x];
                }
            }
        }
        return sum;
    }

    
    /**
     * Sums the amount of "incorrect" items.
     * @return The count of the negatives.
     */
    private int sumNegatives() {
        int sum = 0;
        for(int y = 0; y < 10; ++y) {
            for(int x = 0; x < 10; ++x) {
                if(x != y) {
                    sum += matrix[y][x];
                }
            }
        }
        return sum;
    }


    /**
     * Prints the contents of the confusion matrix.
     */
    public void print() {
        System.out.println();
            for(int y = 0; y < 10; ++y) {
                for(int x = 0; x < 10; ++x) {
                    System.out.print(matrix[y][x] + "\t");
                }
                System.out.println();
            }
    }
}

