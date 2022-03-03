using NeoCortexApi.Entities;
using System;
using System.Collections.Generic;
using System.IO;
using System.Text;
using System.Linq;
using System.Diagnostics;
using NeoCortexApi;
using NeoCortexApi.Encoders;
using NeoCortex;
using NeoCortexApi.Utility;
using System.Text.RegularExpressions;
using NeoCortexApi.Classifiers;
using NeoCortexApi.Network;

namespace SequenceLearningExperiment
{
    public class HelperMethods
    {
        static readonly string[] CancerSequenceClasses = new string[] { "inactive - exp", "mod. active", "very active", "inactive - virtual" };
        static readonly float[][] CancerSequenceClassesOneHotEncoding = new float[][] { new float[] { 0, 0, 0, 1 }, new float[] { 0, 0, 1, 0 }, new float[] { 0, 1, 0, 0 }, new float[] { 1, 0, 0, 0 } };

        /// <summary>
        ///     Fetch Cancer Peptides data from file
        /// </summary>
        /// <param name="dataFilePath"></param>
        /// <returns></returns>
        public static List<Dictionary<string, string>> ReadCancerSequencesDataFromFile(string dataFilePath)
        {
            List<Dictionary<string, string>> SequencesCollection = new List<Dictionary<string, string>>();

            int keyForUniqueIndexes = 0;

            if (File.Exists(dataFilePath))
            {

                using (StreamReader sr = new StreamReader(dataFilePath))
                {
                    while (sr.Peek() >= 0)
                    {
                        var line = sr.ReadLine();
                        string[] values = line.Split(",");



                        Dictionary<string, string> Sequence = new Dictionary<string, string>();

                        string label = values[1];
                        string sequenceString = values[0];

                        int count = 0;

                        foreach (var alphabet in sequenceString)
                        {
                            keyForUniqueIndexes++;
                            if (Sequence.ContainsKey(alphabet.ToString()))
                            {
                                var newKey = alphabet.ToString() + "," + keyForUniqueIndexes;
                                Sequence.Add(newKey, label);
                            }
                            else
                            {
                                Sequence.Add(alphabet.ToString(), label);
                            }
                        }

                        SequencesCollection.Add(Sequence);
                    }

                }

                return SequencesCollection;
            }
            return null;
        }

        /*
        /// <summary>
        /// 
        /// </summary>
        /// <param name="dataFilePath"></param>
        /// <returns></returns>
        public static List<Dictionary<string, int[]>> ReadAndEncodeCancerSequencesDataFromFileV2(string dataFilePath)
        {
            List<Dictionary<string, int[]>> SequencesCollection = new List<Dictionary<string, int[]>>();
            ScalarEncoder encoder_Alphabets = FetchAlphabetEncoder();
            int keyForUniqueIndexes = 0;

            if (File.Exists(dataFilePath))
            {

                using (StreamReader sr = new StreamReader(dataFilePath))
                {
                    while (sr.Peek() >= 0)
                    {
                        var line = sr.ReadLine();
                        string[] values = line.Split(",");


                        string label = values[1];
                        string sequenceString = values[0];
                        Dictionary<string, int[]> tempDictionary = new Dictionary<string, int[]>();

                        if (sequenceString.Length < 33)
                        {
                            int remainingLength = 33 - sequenceString.Length;
                            for (int i = 0; i < remainingLength; i++)
                            {
                                sequenceString = sequenceString + "Z";
                            }
                        }

                        int[] sdr = new int[0];
                        foreach (var alphabet in sequenceString)
                        {
                            sdr = sdr.Concat(encoder_Alphabets.Encode(char.ToUpper(alphabet) - 64)).ToArray();
                        }
                        tempDictionary.Add(label, sdr);
                        SequencesCollection.Add(tempDictionary);
                    }

                }

                return SequencesCollection;
            }
            return null;
        }
        */
    
        //*********************************************************************************************
        // ENCODING TRAINING DATA

        /// <summary>
        ///     Encoding Amino Alphabetic Sequences
        /// </summary>
        /// <param name="trainingData"></param>
        /// <returns></returns>
        public static List<Dictionary<string, int[]>> EncodeCancerSequences(List<Dictionary<string, string>> trainingData)
        {

            List<Dictionary<string, int[]>> ListOfEncodedTrainingSDR = new List<Dictionary<string, int[]>>();

            ScalarEncoder encoder_Alphabets = FetchAlphabetEncoder();

            foreach (var sequence in trainingData)
            {
                int keyForUniqueIndex = 0;
                int index = 0;
                var tempDictionary = new Dictionary<string, int[]>();

                foreach (var element in sequence)
                {
                    keyForUniqueIndex++;
                    var elementLabel = element.Key + "," + element.Value;
                    var elementKey = element.Key;
                    int[] sdr = new int[0];
                    sdr = sdr.Concat(encoder_Alphabets.Encode(char.ToUpper(element.Key.ElementAt(0)) - 64)).ToArray();

                    //UNCOMMENT THESE LINES TO DRAW SDR BITMAP
                    //int[,] twoDimenArray = ArrayUtils.Make2DArray<int>(ElementWiseSdrList[ElementWiseSdrList.Count - 1], 100, 100);
                    //    var twoDimArray = ArrayUtils.Transpose(twoDimenArray
                    // NeoCortexUtils.DrawBitmap(twoDimArray, 1024, 1024, $"{observationLabel.ToString()}.png", null);

                    if (tempDictionary.ContainsKey(elementLabel))
                    {
                        var newKey = elementLabel + "," + keyForUniqueIndex;
                        tempDictionary.Add(newKey, sdr);
                    }
                    else
                    {
                        tempDictionary.Add(elementLabel, sdr);
                    }
                }
                ListOfEncodedTrainingSDR.Add(tempDictionary);
            }
            return ListOfEncodedTrainingSDR;
        }

        /*
        public static Dictionary<string, int[]> EncodeCancerSequencesApproach2(List<Dictionary<string, string>> trainingData)
        {


            var tempDictionary = new Dictionary<string, int[]>();
            ScalarEncoder encoder_Alphabets = FetchAlphabetEncoder();

            foreach (var sequence in trainingData)
            {
                int keyForUniqueIndex = 0;
                int index = 0;

                int[] sdr = new int[0];
                var elementLabel = "";
                foreach (var element in sequence)
                {
                    keyForUniqueIndex++;
                    elementLabel = element.Value.Split("_")[0];
                    var elementKey = element.Key;

                    sdr = sdr.Concat(encoder_Alphabets.Encode(char.ToUpper(element.Key.ElementAt(0)) - 64)).ToArray();
                    //UNCOMMENT THESE LINES TO DRAW SDR BITMAP
                    //int[,] twoDimenArray = ArrayUtils.Make2DArray<int>(ElementWiseSdrList[ElementWiseSdrList.Count - 1], 100, 100);
                    //    var twoDimArray = ArrayUtils.Transpose(twoDimenArray
                    // NeoCortexUtils.DrawBitmap(twoDimArray, 1024, 1024, $"{observationLabel.ToString()}.png", null);


                }

                tempDictionary.Add(elementLabel, sdr);
            }
            return tempDictionary;
        }
        */

        public static Dictionary<float[][], float[][]> CancerSequenceDataProcessing(List<Dictionary<string, string>> trainingData)
        {

            var ListOfProcessedSequenceDictionary = new List<Dictionary<float[][], float[][]>>();


            foreach (var sequence in trainingData)
            {

                var ProcessedSequenceDictionary = new Dictionary<float[][], float[][]>();
                var processedSequence = new float[sequence.Count][];
                var processedLabel = new float[sequence.Count][];
                int elementIndex = 0;

                foreach (var element in sequence)
                {

                    var observationElement = char.ToUpper(element.Value.ElementAt(0)) - 64;
                    var observationLabel = element.Key.Split("_")[0];
                    var observationClassOneHotEncoding = new float[0];

                    if (observationLabel.Equals(CancerSequenceClasses[0]))
                    {
                        observationClassOneHotEncoding = CancerSequenceClassesOneHotEncoding[0];
                    }
                    else if (observationLabel.Equals(CancerSequenceClasses[1]))
                    {
                        observationClassOneHotEncoding = CancerSequenceClassesOneHotEncoding[1];
                    }
                    else if (observationLabel.Equals(CancerSequenceClasses[2]))
                    {
                        observationClassOneHotEncoding = CancerSequenceClassesOneHotEncoding[2];

                    }
                    else if (observationLabel.Equals(CancerSequenceClasses[3]))
                    {
                        observationClassOneHotEncoding = CancerSequenceClassesOneHotEncoding[3];
                    }

                    processedSequence[elementIndex] = new float[] { observationElement };
                    processedLabel[elementIndex] = observationClassOneHotEncoding;

                    ProcessedSequenceDictionary.Add(processedLabel, processedSequence);
                    ListOfProcessedSequenceDictionary.Add(ProcessedSequenceDictionary);

                    elementIndex++;
                }
                return ProcessedSequenceDictionary;
            }
            return null;
        }

        //*********************************************************************************************
        // HTM HELPER METHODS
        
        /// <summary>
        ///     Fetch HTM Config
        /// </summary>
        /// <param name="inputBits"></param>
        /// <param name="numColumns"></param>
        /// <returns></returns>
        public static HtmConfig FetchConfig(int inputBits, int numColumns)
        {
            HtmConfig cfg = new HtmConfig(new int[] { inputBits }, new int[] { numColumns })
            {
                Random = new ThreadSafeRandom(42),

                CellsPerColumn = 32,
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
            return cfg;
        }
        public static string GetKey(List<string> prevInputs)
        {
            string key = String.Empty;

            for (int i = 0; i < prevInputs.Count; i++)
            {
                if (i > 0)
                    key += "-";

                key += (prevInputs[i]);
            }

            return key;
        }
        //**********************************************************************************************

        //*****************************ENCODE USERINPUT FOR TES*****************************************
        
        /// <summary>
        ///         ENCODE USER INPUT 
        /// </summary> 

        /*
        public static int[] EncodeSingleInput_testingExperiment_1(string userInput)
        {

            DateTime date = DateTime.Parse(userInput);
            var day = date.Day;
            var month = date.Month;
            var year = date.Year;
            var weekDay = (int)date.DayOfWeek;

            ScalarEncoder DayEncoder = FetchDayEncoder();
            ScalarEncoder MonthEncoder2 = FetchMonthEncoder();
            ScalarEncoder YearEncoder3 = FetchYearEncoder();
            ScalarEncoder WeekDayEncoder = FetchWeekDayEncoder();

            int[] sdr = new int[0];
            sdr = sdr.Concat(DayEncoder.Encode(day)).ToArray();
            sdr = sdr.Concat(MonthEncoder2.Encode(month)).ToArray();
            sdr = sdr.Concat(YearEncoder3.Encode(year)).ToArray();
            sdr = sdr.Concat(WeekDayEncoder.Encode(weekDay)).ToArray();
            return sdr;
        }
        */

        public static List<int[]> EncodeSingleInput_testingExperiment(string userInput, Boolean EncodeSingleAlphabet)
        {

            var alphabetEncoder = FetchAlphabetEncoder();

            var Encoded_Alphabet_SDRs = new List<int[]>();
            if (!EncodeSingleAlphabet)
            {
                if (userInput.Length < 33)
                {
                    int remainingLength = 33 - userInput.Length;
                    for (int i = 0; i < remainingLength; i++)
                    {
                        userInput = userInput + "Z";
                    }
                }

                foreach (var alphabet in userInput)
                {
                    Encoded_Alphabet_SDRs.Add(alphabetEncoder.Encode(char.ToUpper(alphabet) - 64));
                }
            }
            else
            {
                Encoded_Alphabet_SDRs.Add(alphabetEncoder.Encode(char.ToUpper(userInput.ElementAt(0)) - 64));
            }

            return Encoded_Alphabet_SDRs;
        }

        /*
        public static int[] EncodeSingleInput_testingExperiment_2(string userInput) 
        {
            var alphabetEncoder = FetchAlphabetEncoder();
            var encodedSDR = new int[0];

            foreach (var alphabet in userInput) {
                var encodedElement = alphabetEncoder.Encode(alphabet);
                encodedSDR = encodedSDR.Concat(encodedElement).ToArray();
            }
            return encodedSDR;
        }
        */

        //***********************************************************************************************

        //**********************************FETCHING ENCODERS SECTION****************************************
        
        /// <summary>
        ///         FETCH ENCODERS 
        /// </summary>
        /// <returns> SCALAR ENCODERS</returns>
        /// 

        /*
        public static ScalarEncoder FetchWeekDayEncoder()
        {
            ScalarEncoder MonthEncoder = new ScalarEncoder(new Dictionary<string, object>()
            {
                { "W", 3},
                { "N", 11},
                { "MinVal", (double)0}, // Min value = (0).
                { "MaxVal", (double)7}, // Max value = (7).
                { "Periodic", true},
                { "Name", "WeekDay"},
                { "ClipInput", true},
            });
            return MonthEncoder;
        }
        public static ScalarEncoder FetchDayEncoder()
        {
            ScalarEncoder DayEncoder = new ScalarEncoder(new Dictionary<string, object>()
            {
                { "W", 3},
                { "N", 35},
                { "MinVal", (double)1}, // Min value = (1).
                { "MaxVal", (double)32}, // Max value = (32).
                { "Periodic", true},
                { "Name", "Date"},
                { "ClipInput", true},
           });

            return DayEncoder;
        }
        public static ScalarEncoder FetchMonthEncoder()
        {
            ScalarEncoder MonthEncoder = new ScalarEncoder(new Dictionary<string, object>()
            {
                { "W", 3},
                { "N", 15},
                { "MinVal", (double)1}, // Min value = (1).
                { "MaxVal", (double)12}, // Max value = (12).
                { "Periodic", true},
                { "Name", "Month"},
                { "ClipInput", true},
            });
            return MonthEncoder;
        }
        public static ScalarEncoder FetchYearEncoder()
        {
            ScalarEncoder YearEncoder = new ScalarEncoder(new Dictionary<string, object>()
            {
                { "W", 3},
                { "N", 11},
                { "MinVal", (double)2018}, // Min value = (2018).
                { "MaxVal", (double)2022}, // Max value = (2021).
                { "Periodic", false},
                { "Name", "Year"},
                { "ClipInput", true},
            });
            return YearEncoder;
        }
        */

        public static ScalarEncoder FetchAlphabetEncoder()
        {
            ScalarEncoder AlphabetEncoder = new ScalarEncoder(new Dictionary<string, object>()
                {
                    { "W", 5},
                    { "N", 31},
                    { "Radius", -1.0},
                    { "MinVal", (double)1},
                    { "Periodic", true},
                    { "Name", "scalar"},
                    { "ClipInput", false},
                    { "MaxVal", (double)27}
                });
            return AlphabetEncoder;
        }
        //************************************************************************************************

        //*************************TESTING METHODS*******************************************
        
        /// <summary>
        ///  CANCER SEQUENCE CLASSIFICATION EXPERIMENT
        /// </summary>
        /// <param name="trainingData">TRAINING DATA</param>
        /// <param name="trained_CortexLayer">TRAINED HTM CORTEX LAYER</param>
        /// <param name="trained_Classifier">TRAINED HTM CLASSIFIER</param>
        public static void BeginAutomatedTestingExperiment(List<Dictionary<string, string>> trainingData, CortexLayer<object, object> trained_CortexLayer, HtmClassifier<string, ComputeCycle> trained_Classifier)
        {
            Random rnd = new Random();

            var testingResults = new List<string>();

            var numberOfDataPoints = (int)(0.7 * trainingData.Count);
            int correctPrediction = 0;
            for (int i = 0; i < numberOfDataPoints; i++)
            {
                var index = rnd.Next(0, trainingData.Count - 1);
                var sequence = trainingData[index];
                Dictionary<string, List<string>> predictedInput = new Dictionary<string, List<string>>();

                List<string> possibleClasses = new List<string>();

                for (int j = 0; j < sequence.Count; j++)
                {

                    var element = sequence.ElementAt(j);
                    var elementLabel = element.Value;

                    // WHILE CREATING TRAINING UNIQUE KEY INDEXING PROCESS
                    // INVOLVES INCLUDING "originalKey +,{Number} " , therefore To fetch 
                    // original key we are spliting the element key.
                    // example A,7  F,23 and so on.

                    var elementString = element.Key.Split(",")[0];

                    var elementSDR = EncodeSingleInput_testingExperiment(elementString, true);

                    var lyr_Output = trained_CortexLayer.Compute(elementSDR[0], false) as ComputeCycle;
                    var classifierPrediction = trained_Classifier.GetPredictedInputValues(lyr_Output.PredictiveCells.ToArray(), 5);

                    // CHECK IF INPUT HAS PRODUCED ANY OUTPUTS
                    if (classifierPrediction.Count > 0)
                    {
                        foreach (var prediction in classifierPrediction)
                        {
                            if (j < sequence.Count - 1)
                            {
                                var nextElement = sequence.ElementAt(j + 1);
                                var nextElementString = nextElement.Key.Split(",")[0];
                                if (prediction.PredictedInput.Split(",")[0] == nextElementString)
                                {
                                    possibleClasses.Add(nextElement.Value);
                                }
                            }
                        }

                    }
                }

                var Classcounts = possibleClasses.GroupBy(x => x)
                    .Select(g => new { possibleClass = g.Key, Count = g.Count() })
                    .ToList();
                var possibleClass = "";
                if (Classcounts.Count > 0)
                    possibleClass = Classcounts.Max().possibleClass;

                var elmentClass = sequence.ElementAt(0).Value;
                testingResults.Add($"ELEMENT CLASS :{elmentClass} \t PREDICTED CLASS:{possibleClass}");
                if (possibleClass.Split("_")[0] == elmentClass.Split("_")[0])
                {
                    correctPrediction++;
                }
            }
            
            //************************************PREDICTION ACCURACY************************************************************
            double accuracy = ((double)correctPrediction / numberOfDataPoints) * 100;

            Console.WriteLine($"ACCURACY : {accuracy} \t NumberOfCorrect_Prediction : {correctPrediction} \t TotalDataPoints:{numberOfDataPoints}");
            testingResults.Add($"ACCURACY : {accuracy} \t NumberOfCorrect_Prediction : {correctPrediction} \t TotalDataPoints:{numberOfDataPoints}");

        }
    }
}
