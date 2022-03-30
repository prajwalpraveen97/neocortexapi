# 25. Multi-Sequence/Image Learning Project (WS21/22) 


 **Team Members**
- Harish Palanivel (harish.palanivel@stud.fra-uas.de)
- Gaurav Honnavara Manjunath (gaurav.honnavaramanjunath@stud.fra-uas.de)
- Athkar Praveen Prajwal (praveen.athkar@stud.fra-uas.de)

**Project Description**
=============


1.Objective
-------------

To demonstrate learning of sequences such as set of Number sequences, Alphabets (Cancer Cells Sequences) and Image Data Sets (Apple, Avocado, Banana)


In the previous work, Multi Sequence Learning solution has been implemented for Sequence of Numbers . Our task is to analyse and understand the solution and Develop the MultiSequence Learning Solution for Set of Alphabets (Cancer Cell Sequences) and MultiSequence Learning Solution for Image Data Sets

For Example :
After Training Data Sets, if the user inputs an image such as apple or orange, it has to predict which fruit is identified.


2.Approach (Training & Prediction)
-------------

We introduce different types of Encoders in HtmPredictionEngine such as ScalarEncoder, HTM Image Encoder  for Learning Sequence of Numbers,Learning of Sequence of Alphabets and Image Data Sets.

The Training & Prediction Process includes: 
1. Reading sequences (Sequence of Numbers,Alphabers,Images).
2. Encoding data using encoders(Scalar Encoders - Number,Alphabets ; HTM Image Encoder - Image).
3. Encoded data given as SDR input to Spatial Pooler and train several times until it reaches stable state(Using Homeostatic Plasticity Controller for stability).
4. Prediction of Input Sequence by Comparing with trained data and categorise the data based on observation class(Label) and Accuracy.

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

(i) Input sequence of Numbers to Train Model

[MultiSequenceLearning_Numbers](https://github.com/harishpalani12/neocortexapi/blob/393f4a9e4bdb6d070322db94eecd0ed9490692cf/source/MySEProject/SimpleMultiSequenceLearning/SimpleMultiSequenceLearning/HelperMethod_Numbers.cs#L41-L45)


(ii) Setting Parameters in HTM Configuration and Train Sequence using Scalar Encoder (Which includes Stablity using HomeostaticPlasticityController)

[Training](https://github.com/harishpalani12/neocortexapi/blob/393f4a9e4bdb6d070322db94eecd0ed9490692cf/source/MySEProject/SimpleMultiSequenceLearning/SimpleMultiSequenceLearning/MultisequenceLearning.cs#L29-L304)


(iii) Predition Algoritm for Sequence of Numbers

[Prediction](https://github.com/harishpalani12/neocortexapi/blob/1827c9e27dcccdb5d8242989b911945711b36da5/source/MySEProject/SimpleMultiSequenceLearning/SimpleMultiSequenceLearning/HelperMethod_Numbers.cs#L86-L107)

- **Multi Sequence Learning -Alphabets.**

After we analysed MultiSequence Learning for Sequence of Numbers, we moved further to Train and Predict Sequence of Alphabet.
In This Approach, the sequence of alphabets were stored in .csv file and identified those sequences with different labels. The Solution was modified to read these sequence of Alphabets from .csv file and Train Alphabets by making use of AlphabetsEncoder(Scalar Encoder) and HTM Prediction Algoritm.Prediction algorithm was developed to predict the trained sequences where the similarity matrix generated is compared with each of the SDRs of the Sequence learned during the training phase and based on the accuracy and observation class (Label), the Sequence is predicted

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

(i) Encode Input sequence of Alphabets(from .CSV file)

[ReadDataFromCSV](https://github.com/harishpalani12/neocortexapi/blob/393f4a9e4bdb6d070322db94eecd0ed9490692cf/source/MySEProject/SimpleMultiSequenceLearning/SimpleMultiSequenceLearning/HelperMethod_Alphabets.cs#L40-L80)

[Enode](https://github.com/harishpalani12/neocortexapi/blob/393f4a9e4bdb6d070322db94eecd0ed9490692cf/source/MySEProject/SimpleMultiSequenceLearning/SimpleMultiSequenceLearning/HelperMethod_Alphabets.cs#L88-L120)

(ii) Setting Parameters in HTM Configuration and Train Sequence using Scalar Encoder (Which includes Stablity using HomeostaticPlasticityController)

[Training](https://github.com/harishpalani12/neocortexapi/blob/393f4a9e4bdb6d070322db94eecd0ed9490692cf/source/MySEProject/SimpleMultiSequenceLearning/SimpleMultiSequenceLearning/MultisequenceLearning.cs#L378-L642)

(iii) Predition Algoritm for Sequence of Alphabets

[Prediction](https://github.com/harishpalani12/neocortexapi/blob/393f4a9e4bdb6d070322db94eecd0ed9490692cf/source/MySEProject/SimpleMultiSequenceLearning/SimpleMultiSequenceLearning/HelperMethod_Alphabets.cs#L127-L155)

- **Multi Sequence Learning -Image Data Sets.**

After we analysed MultiSequence Learning for Sequence of Alphabets, we moved further to Train and Predict Image DataSets.
In This Approach, the Image Data Sets are stored in local folder with subfolders categories as labels.
Code is developed to binarize the image data sets and store in the local directory. The HTM Image Encoder binarizes the input image and stores as array elements of zeros and ones used as SDR Input for training.
Prediction of Image algorithm was developed, and the input image was predicted by comparing with the trained data sets and returning the prediction output based on accuracy and Observation class (Label).


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

(i) Encode Input Image Data Sets from 'Input Folder' and Binarizing the Image Data sets

[Encoding & Binarization](https://github.com/harishpalani12/neocortexapi/blob/1827c9e27dcccdb5d8242989b911945711b36da5/source/MySEProject/SimpleMultiSequenceLearning/SimpleMultiSequenceLearning/HelperMethod_Images.cs#L60-L78)

(ii) Setting Parameters in HTM Configuration and Train Sequence using HTM Image Encoder (Which includes Stablity using HomeostaticPlasticityController)

[Training](https://github.com/harishpalani12/neocortexapi/blob/1827c9e27dcccdb5d8242989b911945711b36da5/source/MySEProject/SimpleMultiSequenceLearning/SimpleMultiSequenceLearning/Multiseq_ImageLearning.cs#L37-L322)

(iii) Predition Algoritm for Sequence of Alphabets

[Prediction](https://github.com/harishpalani12/neocortexapi/blob/1827c9e27dcccdb5d8242989b911945711b36da5/source/MySEProject/SimpleMultiSequenceLearning/SimpleMultiSequenceLearning/HelperMethod_Images.cs#L92-L114)

 3.Results
-------------

#### 1.Multi Sequence Learning -Numbers.


(i) The Below Figure shows the training accuracy for a sequence of numbers for five sequences

![image](https://github.com/harishpalani12/neocortexapi/blob/1827c9e27dcccdb5d8242989b911945711b36da5/source/MySEProject/SimpleMultiSequenceLearning/Documentation/Images/Training%20Accuracy%20%E2%80%93%20Sequence%20of%20Numbers.jpg)

(ii)Figure below shows the prediction for the sequence of Numbers for the trained data sequence.

![image](https://github.com/harishpalani12/neocortexapi/blob/1827c9e27dcccdb5d8242989b911945711b36da5/source/MySEProject/SimpleMultiSequenceLearning/Documentation/Images/Prediction%20%E2%80%93%20Sequence%20of%20Numbers.jpg)

#### 2.Multi Sequence Learning -Alphabets.

(i) Figure Shows the training accuracy for a sequence of alphabets (Anticancer Peptide Sequence).

![image](https://github.com/harishpalani12/neocortexapi/blob/1827c9e27dcccdb5d8242989b911945711b36da5/source/MySEProject/SimpleMultiSequenceLearning/Documentation/Images/Training%20Accuracy%20%E2%80%93%20Sequence%20of%20Alphabets.jpg)

(ii) Figure Shows the prediction for a particular sequence that is entered by the user.

![image](https://github.com/harishpalani12/neocortexapi/blob/66d9d6a8a9f00c3d3f88b1acd65af026bd4ce9d8/source/MySEProject/SimpleMultiSequenceLearning/Documentation/Images/Prediction%20%E2%80%93%20Sequence%20of%20Alphabets.jpg)

#### 3.Multi Sequence Learning -Image Data Sets.

(i) Figure Shows the training accuracy for Image Data Sets

![image](https://github.com/harishpalani12/neocortexapi/blob/1827c9e27dcccdb5d8242989b911945711b36da5/source/MySEProject/SimpleMultiSequenceLearning/Documentation/Images/Training%20Accuracy%20%E2%80%93%20Image%20datasets.jpg)

(ii)Figure shows the prediction for the input Images for the trained Image data.

![image](https://github.com/harishpalani12/neocortexapi/blob/1827c9e27dcccdb5d8242989b911945711b36da5/source/MySEProject/SimpleMultiSequenceLearning/Documentation/Images/Prediction%20%E2%80%93%20Image%20Data%20Sets.jpg)

 4.Conclusion
-------------

Multi Sequence learning for Sequence of Numbers which uses Neocortex API is used as a reference model to develop a solution for Multi Sequence learning - Sequence of Alphabets and Multi Sequence learning- Image data sets. 

HTM Prediction Engine was modified with different parameters to match the respective training process. 
The Sequence of Alphabets (Anticancer Peptide Sequence) Stored as a CSV file was modified and stored as an encoded value in the dictionary using Scalar Encoder and SDR input for the Training process. A prediction algorithm was developed to predict the trained sequences where the similarity matrix generated is compared with each of the SDRs of the Sequence learned during the training phase and based on the accuracy and observation class (Label), the Sequence is predicted.

HTM Image Encoder was incorporated to develop a solution that could train multiple Image data sets and a prediction algorithm that could predict input images. The HTM Image Encoder binarizes the input image and stores as array elements of zeros and ones used as SDR Input for training. 
Prediction of Image algorithm was developed, and the input image was predicted by comparing with the trained data sets and returning the prediction output based on accuracy and Observation class (Label).

We performed Multi Sequence Learning for a different sequence of data sets and could achieve up to 87.5% of accuracy in the Training Phase. 

The experiments carried out helped us understand different types of encoders, such as scalar encoders and HTM Image encoders, how the Spatial pooler creates SDR inputs and computes the learning phase, and how the Homeostatic Plasticity controller helps in stabilizing the learning phase in NeoCortex API.


5.References
-------------


[1] Continuous online sequence learning with an unsupervised neural network model. Author: Yuwei Cui, Subutai Ahmad, Jeff Hawkins| Numenta Inc.

[2] On the performance of HTM predictions of Medical Streams in real-time. Author: Noha O. El-Ganainy, Ilangkp Balasingham, Per Steinar Halvorsen, Leiv Arne Rosseland.

[3] Sequence memory for prediction, inference, and behaviour Author: Jeff Hawkins, Dileep George, Jamie Niemasik | Numenta Inc.

[4] An integrated hierarchical temporal memory network for real-time continuous multi interval prediction of data streams Author: Jianhua Diao, Hyunsyug Kang.

[5] Stock Price Prediction Based on Morphological Similarity Clustering and Hierarchical Temporal Memory Author: XINGQI WANG, KAI YANG, TAILIAN LIU
