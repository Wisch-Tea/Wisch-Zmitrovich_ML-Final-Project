import java.io.*;
import java.util.*;
import java.awt.Color;
import java.awt.image.BufferedImage;
import javax.imageio.ImageIO;

/**
 * Responsible for parsing in data from the archive/signs file.
 */
public class ImageParser {
    
    public ImageParser() {}

    /**
     * Creates a new <code>parseImages</code> parser that can take an image's path, convert it to
     * a 2-D matrix based on the pixel RGB data, downsize the image through averaging the 2-D image matrix
     * segments, convert a 2-D matrix to 1-D, and vice-versa convert a 1-D matrix to 2-D.
     *
     * @param imagesFolderPath
     *     The containing folder string path name to be read from.
     */
    public Input[] parseImages(String imagesFolderPath) {
        List<Input> inputList = new ArrayList<>();
        for(int signValue = 0; signValue < 26; ++signValue) {
            File folder = new File(imagesFolderPath + "/" + signValue);
            for(File currentFile : folder.listFiles()) {
                double[] imageData = getArrayFromMatrix(downsizeImage(parseImageFromFile(currentFile), 8));
                inputList.add(new Input(imageData, signValue));
            }
        }
        return inputList.toArray(new Input[inputList.size()]);
    }

    /**
     * Generates a matrix of the .jpg in a file.
     * @param file The file object of the .jpg.
     * @return The matrix of the pixel values.
     */
    private double[][] parseImageFromFile(File file) {
        double[][] imagePixels;
        try {
            BufferedImage image = ImageIO.read(file);
            imagePixels = new double[image.getHeight()][image.getWidth()];
            for(int rowIndex = 0; rowIndex < image.getHeight(); ++rowIndex) {
                for(int columnIndex = 0; columnIndex < image.getWidth(); ++columnIndex) {
                    imagePixels[rowIndex][columnIndex] = getPixelValue(image, rowIndex, columnIndex);
                }
            }
        } catch(Exception e) {
            System.err.println("Error with reading " + file.getName() + ": " + e.getMessage());
            imagePixels = null;
        }
        return imagePixels;
    }

    /**
     * Calculates a scaled value of a pixel from an image.
     * @param image An image object.
     * @param rowIndex The row index of the pixel.
     * @param columnIndex The column index of the pixel.
     * @return The value of the pixel scaled between 0-255;
     */
    private double getPixelValue(BufferedImage image, int rowIndex, int columnIndex) {
        Color color = new Color(image.getRGB(columnIndex, rowIndex));
        // The maximum value of each of the 3 RGB colors is 255.
        // To scale the pixel between 0-255, the total RGB value must be divided by 255*3.
        return (color.getRed() + color.getGreen() + color.getBlue()) / (255 * 3);
    }

    /**
     * Resizes a matrix by taking the average of DOWNSIZE_FACTOR by DOWNSIZE_FACTOR segments.
     * @param imageMatrix The 2-D matrix to be downsized.
     * @return The downsized 2-D matrix.
     */
    public double[][] downsizeImage(double[][] imageMatrix, int downsizeFactor) {
        double[][] downsizedImage = new double[imageMatrix.length / downsizeFactor][imageMatrix[0].length / downsizeFactor];
        for(int rowIndex = 0; rowIndex < imageMatrix.length / downsizeFactor; ++rowIndex) {
            for(int columnIndex = 0; columnIndex < imageMatrix[0].length / downsizeFactor; ++columnIndex) {
                downsizedImage[rowIndex][columnIndex] = getAverageOfImageSegment(imageMatrix, rowIndex, columnIndex, downsizeFactor);
            }
        }
        return downsizedImage;
    }

    /**
     * Calculates the average of the items in a segment of a matrix.
     * @param imageMatrix The 2-D image matrix.
     * @param rowIndex 
     * @param columnIndex
     * @return
     */
    private double getAverageOfImageSegment(double[][] imageMatrix, int rowIndex, int columnIndex, int downsizeFactor) {
        rowIndex *= downsizeFactor;
        columnIndex *= downsizeFactor;
        double sum = 0;
        for(int i = rowIndex; i < rowIndex + downsizeFactor; ++i) {
            for(int j = columnIndex; j < columnIndex + downsizeFactor; ++j) {
                sum += imageMatrix[i][j];
            }
        }
        return sum / (downsizeFactor * downsizeFactor);
    }


    /* 
    * Pass in a 2-D image matrix and convert it to a 1-D array that has the squared value of
    * [n] dimensionality that's taken from the 2-D array's [n x n] dimension values (square
    * the 'n' value from the 2-D matrix/array to get the 1-D array length).
    * @param imageMatrix The 2-D image matrix.
    */
    public double[] getArrayFromMatrix(double[][] imageMatrix) {
        double[] array = new double[imageMatrix.length * imageMatrix[0].length];
        int indexCount = 0;
        for(int rowIndex = 0; rowIndex < imageMatrix.length; ++rowIndex) {
            for(int columnIndex = 0; columnIndex < imageMatrix[0].length; ++columnIndex) {
                array[indexCount] = imageMatrix[rowIndex][columnIndex];
                ++indexCount;
            }
        }
        return array;
    }

    /* 
    * Pass in a 1-D image array and convert it to a 2-D matrix that has the square-root of 
    * the 1-D array's 'n' dimensionality as [n x n] dimensions of the new 2-D matrix (i.e.
    * sqrt(4) would give 2, so the newly created 2-D matrix would have the dimensions [2 x 2]).
    */
    public double[][] getMatrixFromArray(double[] imageArray) {
        int dimension = (int)Math.sqrt(imageArray.length);
        double[][] matrix = new double[dimension][dimension];
        int indexCount = 0;
        for(int rowIndex = 0; rowIndex < dimension; ++rowIndex) {
            for(int columnIndex = 0; columnIndex < dimension; ++columnIndex) {
                matrix[rowIndex][columnIndex] = imageArray[indexCount];
                ++indexCount;
            }
        }
        return matrix;
    }

    
    // Testing method to print the passed in 2-D matrix/array:
    public void printImage(double[][] matrix) {
        for(int rowIndex = 0; rowIndex < matrix.length; ++rowIndex) {
            for(int columnIndex = 0; columnIndex < matrix.length; ++columnIndex) {
                System.out.print(" " + matrix[rowIndex][columnIndex]);
            }
            System.out.println();
        }
    }
}
