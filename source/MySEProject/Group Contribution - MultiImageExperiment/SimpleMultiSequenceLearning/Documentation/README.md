# 25. Multi-Sequence/Image Learning Project (WS21/22) 


 **Team Members**
- Harish Palanivel, 1392283 (harish.palanivel@stud.fra-uas.de)
- Gaurav Honnavara Manjunath, 1384178 (gaurav.honnavaramanjunath@stud.fra-uas.de)
- Athkar Praveen Prajwal, 1394663 (praveen.athkar@stud.fra-uas.de)




**Project Description**
=============


1.Objective
-------------

To demonstrate learning of sequences such as set of Number sequences, Alphabets (Cancer Cells Sequences) and Image Data Sets (Apple, Avocado, Banana)


In the previous work, Multi Sequence Learning solution has been implemented for Sequence of Numbers . Our task is to analyse and understand the solution and Develop the MultiSequence Learning Solution for Set of Alphabets (Cancer Cell Sequences) and MultiSequence Learning Solution for Image Data Sets

For Example :
After Training Data Sets, if the user inputs an image such as apple or orange, it has to predict which fruit is identified.


2.Approach
-------------

In the following Approaches, we introduce different types of Encoders in HtmPredictionEngine such as ScalarEncoder, HTM Image Encoder  for Learning Sequence of Numbers,Learning of Sequence of Alphabets and Image Data Sets.

The learning process includes: 
1. Reading sequences.
2. Encoding data using encoders.
3. Spatial Pooler Learning with Homeostatic Plasticity Controller until reaching a stable state.
4. Learning with Spatial pooler and Temporal memory, conditional exit.
5. Interactive testing section, output classification/prediction from input data.

- **Multi Sequence Learning -Numbers.**

In This Approach, by making use of MultiSequence Learning Solution, we analysed how multisequence prediction algorithm works, with the existing solution we tried to modifiy the code by changing various parameters such as different sequence of numbers that were allowed to train, and the user can input the sequence of numbers that needs to be predicted.
Also, we tried to change Configurations in HTM Prediction Engine.

Example Datarow :

```csharp
            sequences.Add("TwoMultiple", new List<double>(new double[] { 2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0 }));
            sequences.Add("ThreeMultiple", new List<double>(new double[] { 3.0, 6.0, 9.0, 12.0, 15.0, 18.0, 21.0, 24.0 }));
            sequences.Add("FiveMultiple", new List<double>(new double[] { 5.0, 10.0, 15.0, 20.0, 25.0, 30.0, 35.0, 40.0 }));
            sequences.Add("SevenMultiple", new List<double>(new double[] { 7.0, 14.0, 21.0, 28.0, 35.0, 42.0, 49.0 }));
            sequences.Add("ElevenMultiple", new List<double>(new double[] { 11.0, 22.0, 33.0, 44.0 }));
```


###### **DataFormat - [Number Sequence] -> [Sequence Class] Sequences - Multi Sequence **

- **Multi Sequence Learning -Alphabets.**

After we analysed MultiSequence Learning for Sequence of Numbers, we moved further to Train and Predict Sequence of Alphabet.
In This Approach, the sequence of alphabets were stored in .csv file and identified those sequences with different labels. The Solution was modified to read these sequence of Alphabets from .csv file and Train Alphabets by making use of AlphabetsEncoder and HTM Prediction Algoritm.

Example Alphabet Sequence : 

```csharp
FAKALKALLKALKAL,inactive - exp_8
FAKKLAKKLKKLAKKLAKKWKL,mod. active_18
FAKIIAKIAKIAKKIL,inactive - exp_10
FAKKALKALKKL,inactive - exp_11
FAKKFAKKFKKFAKKFAKFAFAF,mod. active_12
FAKKLAKKLAKLL,mod. active_16
FAKKLAKKLKKLAKKLAK,inactive - exp_17
GLFDIIKKIAESF,mod. active_28
GLFDIVKKIAGHIAGSI,inactive - exp_29
ILPWKWPWWPWRR,mod. active_42
FKLAFKLAKKAFL,inactive - exp_43
FKVKFKVKVK, inactive - exp_44
```


###### **[Alphabetic Sequence] -> [Sequence Class] Sequences - Multi Sequence**


- **Multi Sequence Learning -Image Data Sets.**

After we analysed MultiSequence Learning for Sequence of Alphabets, we moved further to Train and Predict Image DataSets.
In This Approach, the Image Data Sets are stored in local folder with subfolders categories as labels.
Code is developed to binarize the image data sets and store in the local directory, also the sequence of image data sets that needs to be given to the HTM Prediction Engine
is being modified with the help of Image Encoder.


![image](https://github.com/harishpalani12/neocortexapi/blob/393f4a9e4bdb6d070322db94eecd0ed9490692cf/source/MySEProject/SimpleMultiSequenceLearning/SimpleMultiSequenceLearning/InputFolder/Apple/Apple_1.jpg)
![image](https://github.com/harishpalani12/neocortexapi/blob/393f4a9e4bdb6d070322db94eecd0ed9490692cf/source/MySEProject/SimpleMultiSequenceLearning/SimpleMultiSequenceLearning/InputFolder/Apple/Apple_2.jpg)
![image](https://github.com/harishpalani12/neocortexapi/blob/393f4a9e4bdb6d070322db94eecd0ed9490692cf/source/MySEProject/SimpleMultiSequenceLearning/SimpleMultiSequenceLearning/InputFolder/Apple/Apple_3.jpg)
![image](https://github.com/harishpalani12/neocortexapi/blob/393f4a9e4bdb6d070322db94eecd0ed9490692cf/source/MySEProject/SimpleMultiSequenceLearning/SimpleMultiSequenceLearning/InputFolder/Apple/Apple_4.jpg)

![image](https://github.com/harishpalani12/neocortexapi/blob/393f4a9e4bdb6d070322db94eecd0ed9490692cf/source/MySEProject/SimpleMultiSequenceLearning/SimpleMultiSequenceLearning/InputFolder/Avocado/Avocado_1.jpg)
![image](https://github.com/harishpalani12/neocortexapi/blob/393f4a9e4bdb6d070322db94eecd0ed9490692cf/source/MySEProject/SimpleMultiSequenceLearning/SimpleMultiSequenceLearning/InputFolder/Avocado/Avocado_2.jpg)
![image](https://github.com/harishpalani12/neocortexapi/blob/393f4a9e4bdb6d070322db94eecd0ed9490692cf/source/MySEProject/SimpleMultiSequenceLearning/SimpleMultiSequenceLearning/InputFolder/Avocado/Avocado_3.jpg)
![image](https://github.com/harishpalani12/neocortexapi/blob/393f4a9e4bdb6d070322db94eecd0ed9490692cf/source/MySEProject/SimpleMultiSequenceLearning/SimpleMultiSequenceLearning/InputFolder/Avocado/Avocado_4.jpg)

![image](https://github.com/harishpalani12/neocortexapi/blob/393f4a9e4bdb6d070322db94eecd0ed9490692cf/source/MySEProject/SimpleMultiSequenceLearning/SimpleMultiSequenceLearning/InputFolder/Banana/Banana_1.jpg)
![image](https://github.com/harishpalani12/neocortexapi/blob/393f4a9e4bdb6d070322db94eecd0ed9490692cf/source/MySEProject/SimpleMultiSequenceLearning/SimpleMultiSequenceLearning/InputFolder/Banana/Banana_2.jpg)
![image](https://github.com/harishpalani12/neocortexapi/blob/393f4a9e4bdb6d070322db94eecd0ed9490692cf/source/MySEProject/SimpleMultiSequenceLearning/SimpleMultiSequenceLearning/InputFolder/Banana/Banana_3.jpg)
![image](https://github.com/harishpalani12/neocortexapi/blob/393f4a9e4bdb6d070322db94eecd0ed9490692cf/source/MySEProject/SimpleMultiSequenceLearning/SimpleMultiSequenceLearning/InputFolder/Banana/Banana_4.jpg)



###### **[Image Data Sets] -> [Image Data Set Class] Sequences - Multi Sequence**


 3.Learning Process
-------------

#### 1.Multi Sequence Learning -Numbers.

(i) Input sequence of Numbers to Train Model

[MultiSequenceLearning_Numbers](https://github.com/harishpalani12/neocortexapi/blob/393f4a9e4bdb6d070322db94eecd0ed9490692cf/source/MySEProject/SimpleMultiSequenceLearning/SimpleMultiSequenceLearning/HelperMethod_Numbers.cs#L41-L45)


(ii) Set Parameters in HTM Configuration and Train Sequence using Scalar Encoder (Which includes Stablity using HomeostaticPlasticityController)

[HTM Prediction Engine Parameters](https://github.com/harishpalani12/neocortexapi/blob/393f4a9e4bdb6d070322db94eecd0ed9490692cf/source/MySEProject/SimpleMultiSequenceLearning/SimpleMultiSequenceLearning/MultisequenceLearning.cs#L29-L60)


#### 2.Multi Sequence Learning -Alphabets.

(i) Input sequence of Alphabets from .csv File to Train Model

[Training Files - Alphabets](https://github.com/harishpalani12/neocortexapi/blob/393f4a9e4bdb6d070322db94eecd0ed9490692cf/source/MySEProject/SimpleMultiSequenceLearning/SimpleMultiSequenceLearning/TrainingFiles/TrainingFile.csv)


[ReadSequencesDataFromCSV](https://github.com/harishpalani12/neocortexapi/blob/393f4a9e4bdb6d070322db94eecd0ed9490692cf/source/MySEProject/SimpleMultiSequenceLearning/SimpleMultiSequenceLearning/HelperMethod_Alphabets.cs#L40-L80)


(ii) Set Parameters in HTM Configuration and Train Sequence using FetchAlphabetEncoder (Which includes Stablity using HomeostaticPlasticityController)

[TrainEncodeSequencesFromCSV](https://github.com/harishpalani12/neocortexapi/blob/393f4a9e4bdb6d070322db94eecd0ed9490692cf/source/MySEProject/SimpleMultiSequenceLearning/SimpleMultiSequenceLearning/HelperMethod_Alphabets.cs#L88-L120)

[RunAlphabetsLearning](https://github.com/harishpalani12/neocortexapi/blob/393f4a9e4bdb6d070322db94eecd0ed9490692cf/source/MySEProject/SimpleMultiSequenceLearning/SimpleMultiSequenceLearning/MultisequenceLearning.cs#L378-L642)


#### 3.Multi Sequence Learning -Image Data Sets.

(i) Input Image Data Sets from Solution Directory Path  

[ReadImageDataSetsFromFolder](https://github.com/harishpalani12/neocortexapi/blob/393f4a9e4bdb6d070322db94eecd0ed9490692cf/source/MySEProject/SimpleMultiSequenceLearning/SimpleMultiSequenceLearning/HelperMethod_Images.cs#L40-L58)


(ii) Binarize Input Image Data Sets and Train Images and prediction 

[BinarizeImageTraining](https://github.com/harishpalani12/neocortexapi/blob/393f4a9e4bdb6d070322db94eecd0ed9490692cf/source/MySEProject/SimpleMultiSequenceLearning/SimpleMultiSequenceLearning/HelperMethod_Images.cs#L60-L108)



 4.Goals Achieved
-------------

1. Analyse and Improve Multi-Sequence Learning - Numbers
2. Modify Existing Multi-Sequence Solution to Incorporate Multi-Sequence Learning for Set of Alphabets (Also Anti Cancer Peptide Cell Sequences)
3. Addition of HTM Image Encoder to Multi-Sequence Learning Solution and test Image Binarization (Enode, Encode and Save )
4. Train Image Data sets using Multi-Sequence Learning making use of HTM prediction Engine and Image Encoder (without checking the stability)
5. Predict Image from the Trained Image data sets 


 5.In-Progress
-------------

 Team is Working on :
 

1. Code Improvisation,Testing and Optimization 
2. Documentation Work


6.References
-------------


[1] Continuous online sequence learning with an unsupervised neural network model. Author: Yuwei Cui, Subutai Ahmad, Jeff Hawkins| Numenta Inc.

[2] On the performance of HTM predictions of Medical Streams in real-time. Author: Noha O. El-Ganainy, Ilangkp Balasingham, Per Steinar Halvorsen, Leiv Arne Rosseland.

[3] Sequence memory for prediction, inference, and behaviour Author: Jeff Hawkins, Dileep George, Jamie Niemasik | Numenta Inc.

[4] An integrated hierarchical temporal memory network for real-time continuous multi interval prediction of data streams Author: Jianhua Diao, Hyunsyug Kang.

[5] Stock Price Prediction Based on Morphological Similarity Clustering and Hierarchical Temporal Memory Author: XINGQI WANG, KAI YANG, TAILIAN LIU
