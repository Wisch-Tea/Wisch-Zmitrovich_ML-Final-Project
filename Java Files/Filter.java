import java.util.*;

/**
 * Responsble for managing a filter.
 * The CNN houses multiple filters.
 */
public class Filter {

    double[][] filterMatrix;


    public Filter(double[][] newFilterMatrix) {
        filterMatrix = newFilterMatrix.clone();
    }


    public double[] applyFilter(double[] input) {
        ImageParser parser = new ImageParser();
        double[][] matrix = parser.getMatrixFromArray(input);
        int filteredMatrixDimension = matrix.length - (filterMatrix.length - 1);
        double[][] filteredMatrix = new double[filteredMatrixDimension][filteredMatrixDimension];
        for(int rowIndex = 0; rowIndex < filteredMatrixDimension; ++rowIndex) {
            for(int columnIndex = 0; columnIndex < filteredMatrixDimension; ++columnIndex) {
                filteredMatrix[rowIndex][columnIndex] = applyFilterToMatrixSegment(matrix, rowIndex, columnIndex);
            }
        }
        return parser.getArrayFromMatrix(filteredMatrix);
    }


    private double applyFilterToMatrixSegment(double[][] matrix, int startRowIndex, int startColumnIndex) {
        double result = 0;
        for(int rowIndex = 0; rowIndex < filterMatrix.length; ++rowIndex) {
            for(int columnIndex = 0; columnIndex < filterMatrix.length; ++columnIndex) {
                result += matrix[rowIndex + startRowIndex][columnIndex + startColumnIndex] * filterMatrix[rowIndex][columnIndex];
            }
        }
        return result;
    }


    public double[][] addPaddingLayer(double[][] matrix) {
        double[][] paddedMatrix = new double[matrix.length + 2][matrix.length + 2];
        for(int rowIndex = 0; rowIndex < matrix.length + 2; ++rowIndex) {
            for(int columnIndex = 0; columnIndex < matrix.length + 2; ++columnIndex) {
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
