# Wisch-Zmitrovich_ML-Final-Project
 Group final project for Machine Learning at PSU Summer 2021.

 Description:
    This program can train one of two neural network types: A Multi-Layer/Deep
    Feedforward Neural Network or a Convoluted Neural Network. They both train
    models using back propagation on ~26,000 images of American Sign Language
    (ASL) hand signs, where there are ~1,000 images per alphabetic character.
    This data is randomized and split 60:40 between training and testing data
    with the expected output/target being a corresponding numerical value associated
    with the numbered folder that image is located in and its order in the alphabet,
    (i.e. A is in folder 0, B is in folder 1, and so on). Each image is initially
    256 x 256 pixels large, but is resized down to 32 x 32 so that the program can
    finish in a more concise time period as well as aiding to cut down on the number
    of input units in the neural network. In a CNN each image is ran through a
    pre-selected number of filters that correlate with horizontal, vertical, and
    diagonal patterns that are used to train the model to make a class determination
    based on that pattern training.

Terminal Usage Instructions:
    Remove old .class files: rm *.class
    Compile with: javac GroupProject.java
    Run with: java GroupProject [ARGUMENTS]
    [ARGUMENTS]: "MLP" for MLP execution, and/or "CNN" for CNN execution.
                 (USAGE EXAMPLE: "java GroupProject MLP CNN" to run both)