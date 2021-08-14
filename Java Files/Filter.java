import java.util.*;

/**
 * Responsble for creating and managing a filter matrix.
 * The CNN can house multiple filters.
 */
public class Filter {

    double[][] filterMatrix;

    /**
     * Creates a new Convolutional Neural Network <code>Filter</code> from a passed in 2-D matrix/array
     * to be used in creating the filter matrix.
     *
     * @param newFilterMatrix
     *     The 2-D Matrix/Array filter matrix.
     */
    public Filter(double[][] newFilterMatrix) {
        filterMatrix = newFilterMatrix.clone();
    }

    // Take a 1-D input layer array and iteratively (with a stride of 1) apply the saved filter matrix to each segment of the image.
    public double[] applyFilter(double[] input) {
        ImageParser parser = new ImageParser();
        double[][] matrix = parser.getMatrixFromArray(input);
        int filteredMatrixDimension = matrix.length - (filterMatrix.length - 1);
        double[][] filteredMatrix = new double[filteredMatrixDimension][filteredMatrixDimension];
        for(int rowIndex = 0; rowIndex < filteredMatrixDimension; ++rowIndex) {
            for(int columnIndex = 0; columnIndex < filteredMatrixDimension; ++columnIndex) { // Single-step apply the filter
                filteredMatrix[rowIndex][columnIndex] = applyFilterToMatrixSegment(matrix, rowIndex, columnIndex);
            }
        }
        return parser.getArrayFromMatrix(filteredMatrix);
    }

    // Segment of an image matrix for the filter matrix to be multiplied against (filtered).
    private double applyFilterToMatrixSegment(double[][] matrix, int startRowIndex, int startColumnIndex) {
        double result = 0;
        for(int rowIndex = 0; rowIndex < filterMatrix.length; ++rowIndex) {
            for(int columnIndex = 0; columnIndex < filterMatrix.length; ++columnIndex) { // Sum the products of matrix multiplication.
                result += matrix[rowIndex + startRowIndex][columnIndex + startColumnIndex] * filterMatrix[rowIndex][columnIndex];
            }
        }
        return result;
    }

    // Pad out the dimensions of a passed in matrix by 1 (i.e., an [NxN] dimensional matrix would become [(N+1)x(N+1)]).
    public double[][] addPaddingLayer(double[][] matrix) {
        double[][] paddedMatrix = new double[matrix.length + 2][matrix.length + 2];
        for(int rowIndex = 0; rowIndex < matrix.length + 2; ++rowIndex) {
            for(int columnIndex = 0; columnIndex < matrix.length + 2; ++columnIndex) { // Adding zeros to the new outside layer.
                if(rowIndex == 0 || rowIndex == matrix.length + 1 || columnIndex == 0 || columnIndex == matrix.length + 1) {
                    paddedMatrix[rowIndex][columnIndex] = 0;
                } else {
                    paddedMatrix[rowIndex][columnIndex] = matrix[rowIndex - 1][columnIndex - 1];
                }
            }
        }
        return paddedMatrix;
    }
}