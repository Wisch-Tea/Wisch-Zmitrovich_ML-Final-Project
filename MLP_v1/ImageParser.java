import java.io.*;
import java.util.*;
import java.awt.Color;
import java.awt.image.BufferedImage;
import javax.imageio.ImageIO;


public class ImageParser {

    private final int DOWNSIZE_FACTOR = 8;

    
    public ImageParser() {}


    public Input[] parseImages(String imagesFolderPath) {
        List<Input> inputList = new ArrayList<>();
        for(int signValue = 0; signValue < 26; ++signValue) {
            File folder = new File(imagesFolderPath + "/" + signValue);
            for(File currentFile : folder.listFiles()) {
                double[] imageData = getArrayFromImage(downsizeImage(parseImageFromFile(currentFile)));
                inputList.add(new Input(imageData, signValue));
            }
        }
        return inputList.toArray(new Input[inputList.size()]);
    }


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
            return imagePixels;
        } catch(Exception e) {
            System.err.println("Error with reading " + file.getName() + ": " + e.getMessage());
            imagePixels = null;
        }
        return imagePixels;
    }


    private double getPixelValue(BufferedImage image, int rowIndex, int columnIndex) {
        Color color = new Color(image.getRGB(columnIndex, rowIndex));
        return (color.getRed() + color.getGreen() + color.getBlue()) / (255 * 3);
    }


    private double[][] downsizeImage(double[][] imageMatrix) {
        double[][] downsizedImage = new double[imageMatrix.length / DOWNSIZE_FACTOR][imageMatrix[0].length / DOWNSIZE_FACTOR];
        for(int rowIndex = 0; rowIndex < imageMatrix.length / DOWNSIZE_FACTOR; ++rowIndex) {
            for(int columnIndex = 0; columnIndex < imageMatrix[0].length / DOWNSIZE_FACTOR; ++columnIndex) {
                downsizedImage[rowIndex][columnIndex] = averageImageSegment(imageMatrix, rowIndex, columnIndex);
            }
        }
        return downsizedImage;
    }


    private double averageImageSegment(double[][] imageMatrix, int rowIndex, int columnIndex) {
        rowIndex *= DOWNSIZE_FACTOR;
        columnIndex *= DOWNSIZE_FACTOR;
        double sum = 0;
        for(int i = rowIndex; i < rowIndex + DOWNSIZE_FACTOR; ++i) {
            for(int j = columnIndex; j < columnIndex + DOWNSIZE_FACTOR; ++j) {
                sum += imageMatrix[i][j];
            }
        }
        return sum / (DOWNSIZE_FACTOR * DOWNSIZE_FACTOR);
    }


    private double[] getArrayFromImage(double[][] imageMatrix) {
        double[] toReturn = new double[imageMatrix.length * imageMatrix[0].length];
        for(int rowIndex = 0; rowIndex < imageMatrix.length; ++rowIndex) {
            for(int columnIndex = 0; columnIndex < imageMatrix[0].length; ++columnIndex) {
                toReturn[rowIndex + columnIndex] = imageMatrix[rowIndex][columnIndex];
            }
        }
        return toReturn;
    }
}
