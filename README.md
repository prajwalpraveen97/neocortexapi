# Thesis: Using HTM to learn time series data as LSTM and comparison of performance.

## Supervision
Prof. Damir Dobric

Referees: Prof. Dr. Pech, Prof. Dr. Nauth

Student: Yash Vyas – 1266490, vyas@stud.fra-uas.de

## 1. Motivation:

In the following experiment we introduce different types of sequence into the HTM and LSTM and then compare performance.

    Experiment 1 - Taxi Passenger Count Prediction.
    Experimtnt 2 - Anti Cancer Peptides Sequence Classification

## 2. Overview:


This project references 
<ul>
    <li>Sequence Learning sample, see [SequenceLearning.cs](https://github.com/ddobric/neocortexapi/tree/master/source/Samples/NeoCortexApiSample). </li>
    <li>Video Learning sample, see [VideoLearning.cs] (https://github.com/ddobric/neocortexapi/blob/SequenceLearning_ToanTruong/Project12_HTMCLAVideoLearning/HTMVideoLearning/HTMVideoLearning/VideoLearning.cs)</li>
</ul>
                        

    Learning process include: 
    1. reading sequences.
    2. encoding the data using encoders.
    3. Spatial Pooler Learning with Homeostatic Plasticity Controller until reaching stable state.
    4. Learning with Spatial pooler and Temporal memory, conditional exit.
    5. Interactive testing section, output classification/prediction from input data.

## 3. Data Format:

Experiment 1 - Taxi Passenger Count Prediction

       DataFormat - [DateTime] -> [PassengerCount]
       Example datarow - 01-01-2021 00:00:00,19276
       Sequences - Single Sequence. [Here dates are taken in ascending order as sequence]
       Explaination: 01-01-2021,02-01-2020,...,31-01-2020,01-02-2020,.....ENDING-DATE.
                     EachDate is a element in the sequence.
       
Experiment 2 - Anti Cancer Peptides Sequence Classification

       DataFormat - [Alphabetic Sequence] -> [Sequence Class]
       Sequences  - Multi Sequence - Here Alphabetic Sequence is considered sequnce of characters.
       Example datarow - AAWKWAWAKKWAKAKKWAKAA,mod. active

## 4. Learning Process:
<ul>
    <li>For both the experiment same configuration has been used.</li>
    <li>Current HTM Configuration:</li>
</ul>


```csharp
 HtmConfig cfg = new HtmConfig(new int[] { inputBits }, new int[] { numColumns })
            {
                Random = new ThreadSafeRandom(42),

                CellsPerColumn = 32, // Config 2 => 25 
                GlobalInhibition = true,
                LocalAreaDensity = -1,
                NumActiveColumnsPerInhArea = 0.02 * numColumns,
                PotentialRadius = 65,
                InhibitionRadius = 15,

                MaxBoost = 10.0,
                DutyCyclePeriod = 25,
                MinPctOverlapDutyCycles = 0.75,
                MaxSynapsesPerSegment = 128,

                ActivationThreshold = 15,
                ConnectedPermanence = 0.5,

                // Learning is slower than forgetting in this case.
                PermanenceDecrement = 0.25,
                PermanenceIncrement = 0.15,

                // Used by punishing of segments.
                PredictedSegmentDecrement = 0.1


            };
```

### 1. SP Learning with HomeoStatic Plasticity Controller (HPA):

This first section of learning use Homeostatic Plasticity Controller:
```csharp
HomeostaticPlasticityController hpa = new HomeostaticPlasticityController(mem, Sequences.Count, (isStable, numPatterns, actColAvg, seenInputs) =>
            {
                if (isStable)
                    // Event should be fired when entering the stable state.
                    Console.WriteLine($"STABLE: Patterns: {numPatterns}, Inputs: {seenInputs}, iteration: {seenInputs / numPatterns}");
                else
                    // Ideal SP should never enter unstable state after stable state.
                    Console.WriteLine($"INSTABLE: Patterns: {numPatterns}, Inputs: {seenInputs}, iteration: {seenInputs / numPatterns}");

                // We are not learning in instable state.
                learn = isInStableState = isStable;

                // Clear all learned patterns in the classifier.
                //cls.ClearState();

            }, numOfCyclesToWaitOnChange: 30);
```
### 2. SP+TM Learning:

This is second phase of training, SP+TM are trained together after SP training
```csharp
foreach (var sequence in Sequences)  // SEQUENCE LOOP
            {
                int SequencesMatchCount = 0; // NUMBER OF MATCHES
                var tempLOGFILE = new Dictionary<int, string>();
                int MatchesCount = 0;
                double SaturatedAccuracyCount = 0;

                for (int i = 0; i < maxCycles; i++) // MAXCYCLE LOOP 
                {
                    cycle++;

                    List<string> ElementWisePrediction = new List<string>();
                    List<string> ElementWiseClasses = new List<string>();

                    // ELEMENT IN SEQUENCE MATCHES COUNT
                    int ElementMatches = 0;

                    foreach (var Elements in sequence) // SEQUENCE DICTIONARY LOOP
                    {
                        // OBSERVATION LABEl
                        var observationLabel = Elements.Key;
                        // ELEMENT SDR LIST FOR A SINGLE SEQUENCE
                        var ElementSdr = Elements.Value;

                        List<Cell> actCells = new List<Cell>();
                        var lyrOut = new ComputeCycle();

                        lyrOut = layer1.Compute(ElementSdr, learn) as ComputeCycle;
                        Debug.WriteLine(string.Join(',', lyrOut.ActivColumnIndicies));

                        // Active Cells
                        actCells = (lyrOut.ActiveCells.Count == lyrOut.WinnerCells.Count) ? lyrOut.ActiveCells : lyrOut.WinnerCells;

                        if (!classVotingEnabled)
                        {
                            cls.Learn(observationLabel, actCells.ToArray());

                            if (lastPredictedValue == observationLabel && lastPredictedValue != "")
                            {
                                ElementMatches++;
                                Debug.WriteLine($"Match. Actual value: {observationLabel} - Predicted value: {lastPredictedValue}");
                            }
                            else
                            {
                                Debug.WriteLine($"Mismatch! Actual value: {observationLabel} - Predicted values: {lastPredictedValue}");
                            }


                        }

                        else
                        {
                            cls.Learn(HelperMethods.processLabel(observationLabel), actCells.ToArray());
                        }
```
## 5. Experiment Details:

    In the following section experiments performed are explained in detail with training and testing phase results.

### Taxi Passenger Count Prediction Experiment:

#### Data Preparation
<ul>
<li>Monthly dataset are available at https://www1.nyc.gov/site/tlc/about/tlc-trip-record-data.page. </li>
<li>We have used six month dataset i.e Jan 2021 - June 2021. </li>
<li>PassengerCount is aggregated for each day.</li>
<li>We are fetching data from TrainingFile Directory using. </li>
</ul>

```csharp
public static List<Dictionary<string, List<string>>> ReadPassengerDataFromFile(string dataFilePath)
```
see,[HelperMethods.cs](https://github.com/UniversityOfAppliedSciencesFrankfurt/thesis-LSTM0-vs-HTM-Yash-Vyas/blob/main/ThesisExperiments/ThesisExperiments/HelperMethods.cs).          

#### DataEncoding
For encoding month-date-year, we have used scalar encoders with following configurations.

##### DateEncoder
```csharp
 ScalarEncoder DayEncoder = new ScalarEncoder(new Dictionary<string, object>()
            {
                { "W", 3},
                { "N", 35},
                { "MinVal", (double)1}, // Min value = (0).
                { "MaxVal", (double)32}, // Max value = (7).
                { "Periodic", true},
                { "Name", "Date"},
                { "ClipInput", true},
            });
```
##### MonthEncoder
```csharp
 ScalarEncoder MonthEncoder2 = new ScalarEncoder(new Dictionary<string, object>()
            {
                { "W", 3},
                { "N", 15},
                { "MinVal", (double)1}, // Min value = (0).
                { "MaxVal", (double)12}, // Max value = (7).
                { "Periodic", true}, // Since Monday would repeat again.
                { "Name", "Month"},
                { "ClipInput", true},
            });
```
##### YearEncoder
```csharp
ScalarEncoder YearEncoder3 = new ScalarEncoder(new Dictionary<string, object>()
            {
                { "W", 3},
                { "N", 11},
                { "MinVal", (double)2018}, // Min value = (0).
                { "MaxVal", (double)2022}, // Max value = (7).
                { "Periodic", true}, // Since Monday would repeat again.
                { "Name", "Year"},
                { "ClipInput", true},
            });
```

#### Example of data encoding:
        
<ul>
<li>Raw Data Row : 01-01-2021 00:00:00 [DD-MM-YYYY]</li>
<li>Encoded Data Row : 11000000000000000000000000000000001-110000000000001-00000001110</li>
</ul>

#### Tesing Modes:
![Testing Mode After Learning](https://github.com/UniversityOfAppliedSciencesFrankfurt/thesis-LSTM0-vs-HTM-Yash-Vyas/blob/main/ThesisExperiments/ThesisExperiments/OutPutSnaps/PassengerCount_trained_testingModes.png)  
### 2. Anti Cancer Peptide Sequence Classification:
#### Data Preparation and Processing
<ul>
<li>Dataset are available at https://archive.ics.uci.edu/ml/datasets/Anticancer+peptides. </li>
<li>We are fetching data from TrainingFile Directory using.</li>
<li> We are using elementwise prediction and later applying majority votes value as classification/label value.</li>
</ul>

```csharp
public static List<Dictionary<string, List<string>>> ReadCancerSequencesDataFromFile(string dataFilePath)
```
see,[HelperMethods.cs](https://github.com/UniversityOfAppliedSciencesFrankfurt/thesis-LSTM0-vs-HTM-Yash-Vyas/blob/main/ThesisExperiments/ThesisExperiments/HelperMethods.cs).         

#### DataEncoding
we encode data using scalar encoder for converting alphabet numeric value. 

#### AlphabetEncoder
```csharp
 Dictionary<string, object> settingsScalarEncoder_Alphabets = new Dictionary<string, object>()
                {
                    { "W", 5},
                    { "N", 31},
                    { "Radius", -1.0},
                    { "MinVal", (double)1},
                    { "Periodic", true},
                    { "Name", "scalar"},
                    { "ClipInput", false},
                    { "MaxVal", (double)27}
                };
            ScalarEncoder encoder_Alphabets = new ScalarEncoder(settingsScalarEncoder_Alphabets);
```

#### Example of data encoding:

<ul>
<li>Raw Data Row : AAWKWAWAKKWAKAKKWAKAA [Each alphabet will be encoded seprately]</li>
<li>Encoded Data Row :  A 1:1:1:0:0:0:0:0:0:0:0:0:0:0:0:0:0:0:0:0:0:0:0:0:0:0:0:0:0:1:1</li>
<li>W 0:0:0:0:0:1:1:1:1:1:0:0:0:0:0:0:0:0:0:0:0:0:0:0:0:0:0:0:0:0:0</li>
<li>K 0:0:0:0:0:0:0:0:0:0:1:1:1:1:1:0:0:0:0:0:0:0:0:0:0:0:0:0:0:0:0</li>
<li>....</li>
</ul>

#### Tesing Modes:

#### Version 1 :
Explaination:
Input : FKVKFKVKVK, inactive - exp_44 
Learning : (F_inactive - exp_44)-(K_inactive - exp_44 )...-(K_inactive - exp_44)
Prediction : F-> k,...
             k-> v,...             
![Testing Mode After Learning](https://github.com/UniversityOfAppliedSciencesFrankfurt/thesis-LSTM0-vs-HTM-Yash-Vyas/blob/main/ThesisExperiments/ThesisExperiments/OutPutSnaps/CancerSequenceTrainingModes.png)  

#### Version 2 :
Explaination:
Input : FKVKFKVKVK, inactive - exp_44
Learning: (FKVKFKVKVK_inactive - exp_44)
Note: We are using the whole sequence as a single element
![Testing Mode After Learning](https://github.com/UniversityOfAppliedSciencesFrankfurt/thesis-LSTM0-vs-HTM-Yash-Vyas/blob/main/ThesisExperiments/ThesisExperiments/OutPutSnaps/CancerSequenceClassification_V2_tesingMode.png)  



## Similar Studies/Research used as References
[1] Continuous online sequence learning with an unsupervised neural network model.
Author: Yuwei Cui, Subutai Ahmad, Jeff Hawkins| Numenta Inc.

[2] On the performance of HTM predicions of Medical Streams in real time.
Author: Noha O. El-Ganainy, Ilangkp Balasingham, Per Steinar Halvorsen, Leiv Arne Rosseland.

[3] Sequence memory for prediction, inference and behaviour
Author: Jeff Hawkins, Dileep George, Jamie Niemasik | Numenta Inc.

[4] An integrated hierarchical temporal memory network for real-time continuous multi interval 
prediction of data streams
Author: Jianhua Diao, Hyunsyug Kang.

[5] Stock Price Prediction Based on Morphological Similarity Clustering and Hierarchical Temporal 
Memory
Author: XINGQI WANG, KAI YANG, TAILIAN LIU

Similar Thesis used as References:
[6] Real-time Traffic Flow Prediction using Augmented Reality
Author: Minxuan Zhang
