import java.util.*;
import java.io.*;


public class InputsList extends LayerList {
    /**
     * Default constructor.
     */
    public InputsList() {
        super();
    }


    /**
     * Parameterized constructor.
     * @param imageFileName The name of the file containing the pixel data for the images.
     * @param labelFileName The name of the file containing the labels for the images.
     */
    public InputsList(String imageFileName, String labelFileName) {
        extractData(imageFileName, labelFileName);
    }


    /**
     * Extracts the data from the image and label files.
     * @param imageFileName The name of the file containing the pixel data for the images.
     * @param labelFileName The name of the file containing the labels for the images.
     */
    private void extractData(String imageFileName, String labelFileName) {
        try {
            DataInputStream imageInput = new DataInputStream(new FileInputStream(imageFileName));
            DataInputStream labelInput = new DataInputStream(new FileInputStream(labelFileName));

            int imageMagicNumber = imageInput.readInt();
            int imageNumberOfImages = imageInput.readInt();
            int imageNumberOfRows = imageInput.readInt();
            int imageNumberOfColumns = imageInput.readInt();

            int labelMagicNumber = labelInput.readInt();
            int labelNumberOfImages = labelInput.readInt();

            constructInputs(imageInput, labelInput, imageNumberOfImages, imageNumberOfRows * imageNumberOfColumns);

            imageInput.close();
            labelInput.close();
        } catch (FileNotFoundException e) {
            System.out.println("Error reading file: " + e.getMessage());
            e.printStackTrace();
        } catch (IOException e) {
            System.out.println("Error reading file: " + e.getMessage());
            e.printStackTrace();
        }
    }


    /**
     * Constructs the list of inputs based on the extracted data.
     * @param imageInput The input stream of the file containing the pixel data for the images.
     * @param labelInput The input stream of the file containing the labels for the images.
     * @param numberOfImages The number of images in the file.
     * @param dimension The dimension of the images (will be 28x28 = 784).
     */
    private void constructInputs(DataInputStream imageInput, DataInputStream labelInput, int numberOfImages, int dimension) {
        if(layerList == null) {
            layerList = new Inputs[numberOfImages];
        }
        try {
            for(int i = 0; i < numberOfImages; ++i) {
                Vector<Integer> inputArray = new Vector<>(dimension);
                for(int j = 0; j < dimension; ++j) {
                    int pixelData = imageInput.read();
                    inputArray.add(pixelData);
                }
                int labelData = labelInput.read();
                layerList[i] = new Inputs(resize(inputArray), labelData);
            }
        } catch (FileNotFoundException e) {
            System.out.println("Error reading file: " + e.getMessage());
            e.printStackTrace();
        } catch (IOException e) {
            System.out.println("Error reading file: " + e.getMessage());
            e.printStackTrace();
        }
    }


    /**
     * Resizes the image data of 0-255 pixel values to 0-1 values pixel.
     * @param toResize A Vector of the image pixels to resize.
     * @return The Vector of the resized image pixels.
     */
    private Vector<Double> resize(Vector<Integer> toResize) {
        Vector<Double> toReturn = new Vector<>(toResize.size());
        for(int i = 0; i < toResize.size(); ++i) {
            toReturn.add((double)toResize.get(i) / 255.0);
        }
        return toReturn;
    }


    /**
     * Retrieves an Inputs item from the list.
     * @param index The index from which to retrieve from.
     * @return The Inputs object from the specified index.
     */
    public Inputs getItem(int index) {
        return (Inputs)layerList[index];
    }
}