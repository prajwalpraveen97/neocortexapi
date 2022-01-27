using NeoCortexApi;
using NeoCortexApi.Classifiers;
using NeoCortexApi.Entities;
using NeoCortexApi.Network;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;

namespace SequenceLearningExperiment
{
    class SequenceLearningHTM
    {
        static readonly int numColumns = 2048;

        /// <summary>
        /// TRAINING FILE PATH
        /// </summary>
        static readonly string PassengerCountDataFile_MINI = Path.GetFullPath(System.AppDomain.CurrentDomain.BaseDirectory + @"\TrainingFiles\TaxiPassengerCountPrediction\TrainingFile_MINI.csv");
        static readonly string PassengerCountDataFile_FULL = Path.GetFullPath(System.AppDomain.CurrentDomain.BaseDirectory + @"\TrainingFiles\TaxiPassengerCountPrediction\TrainingFile_FULL.csv");

        static readonly string CancerSequenceDataFile = Path.GetFullPath(System.AppDomain.CurrentDomain.BaseDirectory + @"\TrainingFiles\CancerSequenceClassification\BreastCancer_trainingFile_MINI.csv");
        static readonly string CancerSequenceDataFile2 = Path.GetFullPath(System.AppDomain.CurrentDomain.BaseDirectory + @"\TrainingFiles\CancerSequenceClassification\LungCancer_trainingFile.csv");
        
        /// <summary>
        ///     PASSENGERCOUNT PREDICTION EXPERIMENT ENTRY POINT
        /// </summary>
        public void InitiatePassengerCountPredictionExperiment()
        {
            int inputBits = 72;
            int maxCycles = 15;
            //var trainingData = HelperMethods.ReadPassengerDataFromFile(PassengerCountDataFile_FULL);
            var trainingData = HelperMethods.ReadPassengerDataFromFile(PassengerCountDataFile_MINI);
            var trainingDataProcessed = HelperMethods.EncodePassengerData(trainingData);

            var trained_HTM_model = Run(inputBits,maxCycles,numColumns, trainingDataProcessed, false);
            var trained_CortexLayer = trained_HTM_model.Keys.ElementAt(0);
            var trained_Classifier = trained_HTM_model.Values.ElementAt(0);

            Debug.WriteLine("TESTING TRAINED HTM MODEL ON USERINPUT || TAXI PASSENGER COUNT PREDICTION EXPERIMENT");
            Console.WriteLine("TESTING TRAINED HTM MODEL ON USERINPUT || TAXI PASSENGER COUNT PREDICTION EXPERIMENT ");

            Debug.WriteLine("PLEASE SELECT MODE OF TESTING 1) MANUAL 2) AUTOMATED :");
            Console.WriteLine("PLEASE SELECT MODE OF TESTING 1) MANUAL 2) AUTOMATED :");
            
            
            var testChoice = Console.ReadLine();


            if (testChoice == "1") {
                Debug.WriteLine("PLEASE ENTER DATE FOR PREDICTING PASSENGER COUNT:      *note format->dd-mm-yyyy");
                Console.WriteLine("PLEASE ENTER DATE FOR PREDICTING PASSENGER COUNT:      *note format->dd-mm-yyyy");
                var userInput = Console.ReadLine();
                while (!userInput.Equals("q") && userInput != "Q")
                {
                    var sdr = HelperMethods.EncodeSingleInput_testingExperiment_1(userInput);
                    var userLayerOutput = trained_CortexLayer.Compute(sdr, false) as ComputeCycle;
                    var predictedValuesForUserInput = trained_Classifier.GetPredictedInputValues(userLayerOutput.PredictiveCells.ToArray(), 5);
                    foreach (var predictedVal in predictedValuesForUserInput)
                    {
                        Console.WriteLine("SIMILARITY " + predictedVal.Similarity + " PREDICTED VALUE :" + predictedVal.PredictedInput);
                    }
                    Console.WriteLine("TAKING USERINPUT FOR CHECKING PREDICTED PASSENGER COUNT");
                    userInput = Console.ReadLine();
                }
            }
            else if (testChoice=="2") {
                    HelperMethods.BeginAutomatedTestingExperiment_1(trainingData, trained_CortexLayer, trained_Classifier);
                }
            }

        /// <summary>
        /// Cancer Sequence Classification Experiment EntryPoint
        /// V1:- In the following version we are learining sequence element by element and while prediction trying to predict next element,
        ///      label of element will be used to classify sequence.
        /// </summary>
        public void InitiateCancerSequenceClassificationExperiment()
        {
            int inputBits = 31;
            int maxCycles = 30;
            var trainingData = HelperMethods.ReadCancerSequencesDataFromFile(CancerSequenceDataFile);
            var trainingDataProcessed = HelperMethods.EncodeCancerSequences(trainingData);
            var trained_HTM_model = Run(inputBits, maxCycles, numColumns, trainingDataProcessed, true);
            var trained_CortexLayer = trained_HTM_model.Keys.ElementAt(0);
            var trained_Classifier = trained_HTM_model.Values.ElementAt(0);

            Debug.WriteLine("TESTING TRAINED HTM MODEL ON USERINPUT || CANCER SEQUENCE CLASSIFICATION EXPERIMENT");
            Console.WriteLine("TESTING TRAINED HTM MODEL ON USERINPUT || CANCER SEQUENCE CLASSIFICATION EXPERIMENT");

            Debug.WriteLine("PLEASE SELECT MODE OF TESTING 1) MANUAL 2) AUTOMATED :");
            Console.WriteLine("PLEASE SELECT MODE OF TESTING 1) MANUAL 2) AUTOMATED :");


            var testChoice = Console.ReadLine();
            if (testChoice == "1")
            {
                Debug.WriteLine("PLEASE ENTER CANCER SEQUENCE:             *note format->AAAAVVV {AlphabeticSequence}");
                Console.WriteLine("PLEASE ENTER CANCER SEQUENCE:            *note format->AAAAVVV {AlphabeticSequence}");
                var userInput = Console.ReadLine();
                while (!userInput.Equals("q") && userInput != "Q")
                {
                    var ElementSDRs = HelperMethods.EncodeSingleInput_testingExperiment_2(userInput,false);
                    List<string> possibleClasses = new List<string>();

                    for (int i = 0; i < userInput.Length; i++) {

                        var element = userInput.ElementAt(i);
                        var elementSDR = HelperMethods.EncodeSingleInput_testingExperiment_2(element.ToString(), true);

                        var lyr_Output = trained_CortexLayer.Compute(elementSDR[0], false) as ComputeCycle;
                        var classifierPrediction = trained_Classifier.GetPredictedInputValues(lyr_Output.PredictiveCells.ToArray(), 5);

                        if (classifierPrediction.Count > 0)
                        {

                            foreach (var prediction in classifierPrediction)
                            {
                                if (i < userInput.Length - 1)
                                {
                                    var nextElement = userInput.ElementAt(i + 1).ToString();
                                    var nextElementString = nextElement.Split(",")[0];
                                    if (prediction.PredictedInput.Split(",")[0] == nextElementString) {
                                        if (prediction.PredictedInput.Split(",").Length == 3)
                                        {
                                            {
                                                possibleClasses.Add(prediction.PredictedInput.Split(",")[2]);
                                            }
                                        }
                                        else if (prediction.PredictedInput.Split(",")[0] == nextElementString)
                                        {
                                            possibleClasses.Add(prediction.PredictedInput.Split(",")[1]);
                                        }
                                    }
                                }
                            }

                        }

                    }

                    var Classcounts = possibleClasses.GroupBy(x => x.Split("_")[0])
                   .Select(g => new { possibleClass = g.Key, Count = g.Count() })
                   .ToList();
                    var possibleClass = "";
                    //if (Classcounts.Count > 0)
                    //    possibleClass = Classcounts.Max().possibleClass;
                    foreach(var class_ in Classcounts){
                        Console.WriteLine($"Predicted Class : {class_.possibleClass.Split("_")[0]} \t votes: {class_.Count}");
                    }
                    
                    
                    Console.WriteLine("PLEASE ENTER NEXT SEQUENCE :");
                    userInput = Console.ReadLine();

                }
            }
            else if (testChoice == "2")
            {
                HelperMethods.BeginAutomatedTestingExperiment_2(trainingData, trained_CortexLayer, trained_Classifier);
            }
        }
        /// <summary>
        /// Cancer Sequence Classification Experiment EntryPoint
        /// V2 :- In the following version we are learning sequence as a whole i.e we are taking as single element.
        /// </summary>
        public void InitiateCancerSequenceClassificationExperimentV2()
        {
            int inputBits = 1023;
            int maxCycles = 30;
            var trainingData = HelperMethods.ReadAndEncodeCancerSequencesDataFromFileV2(CancerSequenceDataFile);
            
            var trained_HTM_model = Run(inputBits, maxCycles, numColumns, trainingData, false);
            var trained_CortexLayer = trained_HTM_model.Keys.ElementAt(0);
            var trained_Classifier = trained_HTM_model.Values.ElementAt(0);

            Debug.WriteLine("TESTING TRAINED HTM MODEL ON USERINPUT || CANCER SEQUENCE CLASSIFICATION EXPERIMENT");
            Console.WriteLine("TESTING TRAINED HTM MODEL ON USERINPUT || CANCER SEQUENCE CLASSIFICATION EXPERIMENT");

            Debug.WriteLine("PLEASE SELECT MODE OF TESTING 1) MANUAL 2) AUTOMATED :");
            Console.WriteLine("PLEASE SELECT MODE OF TESTING 1) MANUAL 2) AUTOMATED :");


            var testChoice = Console.ReadLine();
            if (testChoice == "1")
            {
                Debug.WriteLine("PLEASE ENTER CANCER SEQUENCE FOR CLSSIFICATION     **format->ABCSC {without spaces}");
                Console.WriteLine("PLEASE ENTER CANCER SEQUENCE FOR CLSSIFICATION     **format->ABCSC {without spaces}");
                var userInput = Console.ReadLine();
                while (!userInput.Equals("q") && userInput != "Q")
                {
                    var sdr = HelperMethods.EncodeSingleInput_testingExperiment_2(userInput, false);
                    int[] concatedSDR = new int[0];
                    foreach (var elementSDR in sdr) {
                        concatedSDR = concatedSDR.Concat(elementSDR).ToArray();
                    }
                    var predictionList = new List<List<string>>();
                    var sequence = userInput;
                    Dictionary<string, List<string>> predictedInput = new Dictionary<string, List<string>>();

                    List<string> possibleClasses = new List<string>();

                    var lyr_Output = trained_CortexLayer.Compute(concatedSDR, false) as ComputeCycle;
                    var classifierPrediction = trained_Classifier.GetPredictedInputValues(lyr_Output.PredictiveCells.ToArray(), 5);

                    if (classifierPrediction.Count > 0) {
                        foreach (var prediction in classifierPrediction) {
                            Console.WriteLine($"Predicted Input :{prediction.PredictedInput} \t ");
                        }
                    }

                    //    // CHECK IF INPUT HAS PRODUCED ANY OUTPUTS
                    //    if (classifierPrediction.Count > 0)
                    //    {
                    //        foreach (var prediction in classifierPrediction)
                    //        {
                    //            if (j < sequence.Length - 1)
                    //            {
                    //                var nextElement = sequence.ElementAt(j + 1);
                    //                if (prediction.PredictedInput.Split(",")[0] == nextElement.ToString())
                    //                {
                    //                    possibleClasses.Add(prediction.PredictedInput);
                    //                }
                    //            }
                    //        }

                    //    }
                    //}

                    //var Classcounts = possibleClasses.GroupBy(x => x)
                    //    .Select(g => new { possibleClass = g.Key, Count = g.Count() })
                    //    .ToList();
                    //var possibleClass = "";
                    //if (Classcounts.Count > 0)
                    //    possibleClass = Classcounts.Max().possibleClass;

                    ////var elmentClass = sequence.ElementAt(0).Value;
                    ////testingResults.Add($"ELEMENT CLASS :{elmentClass} \t PREDICTED CLASS:{possibleClass}");
                    ////if (possibleClass.Split("_")[0] == elmentClass.Split("_")[0])
                    ////{
                    ////    correctPrediction++;
                    ////}

                    Console.WriteLine("PLEASE ENTER NEXT SEQUENCE :");
                    userInput = Console.ReadLine();

                }
            }
            else if (testChoice == "2")
            {
                //HelperMethods.BeginAutomatedTestingExperimentV2(trainingData, trained_CortexLayer, trained_Classifier);
            }
        }

        /// <summary>
        ///     Run Experiment
        /// </summary>
        /// <param name="inputBits">InputBits Data</param>
        /// <param name="numColumns">NumColumns in Network</param>
        /// <param name="Sequences">Data Sequences</param>
        public Dictionary<CortexLayer<object,object>,HtmClassifier<string,ComputeCycle>> Run(int inputBits,int maxCycles, int numColumns, List<Dictionary<string, int[]>> Sequences, Boolean classVotingEnabled)
        {
            //-----------HTM CONFG
            var htmConfig = HelperMethods.FetchConfig(inputBits, numColumns);
            //--------- CONNECTIONS
            var mem = new Connections(htmConfig);

            // HTM CLASSIFIER
            HtmClassifier<string, ComputeCycle> cls = new HtmClassifier<string, ComputeCycle>();
            // CORTEX LAYER
            CortexLayer<object, object> layer1 = new CortexLayer<object, object>("L1");

            // HPA IS_IN_STABLE STATE FLAG
            bool isInStableState = false;
            // LEARNING ACTIVATION FLAG
            bool learn = true;

            // NUMBER OF NEW BORN CYCLES
            int newbornCycle = 0;

            var OUTPUT_LOG_LIST = new List<Dictionary<int, string>>();
            var OUTPUT_LOG = new Dictionary<int, string>();
            var OUTPUT_trainingAccuracy_graph = new List<Dictionary<int, double>>();
            // HOMOSTATICPLASTICITY CONTROLLER
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

            // SPATIAL POOLER initialization with HomoPlassiticityController using connections.
            SpatialPoolerMT sp = new SpatialPoolerMT(hpa);
            sp.Init(mem);

            // TEMPORAL MEMORY initialization using connections.
            TemporalMemory tm = new TemporalMemory();
            tm.Init(mem);

            // ADDING SPATIAL POOLER TO CORTEX LAYER
            layer1.HtmModules.Add("sp", sp);

            // CONTRAINER FOR Previous Active Columns
            int[] prevActiveCols = new int[0];

            // Starting experiment
            Stopwatch sw = new Stopwatch();
            sw.Start();

            // TRAINING SP till STATBLE STATE IS ACHIEVED
            while (isInStableState == false) // STABLE CONDITION LOOP ::: LOOP - 0
            {
                newbornCycle++;
                Console.WriteLine($"-------------- Newborn Cycle {newbornCycle} ---------------");

                foreach (var sequence in Sequences) // FOR EACH SEQUENCE IN SEQUNECS LOOP ::: LOOP - 1
                {
                    foreach (var Element in sequence) // FOR EACH dictionary containing single sequence Details LOOP ::: LOOP - 2
                    {
                        var observationClass = Element.Key; // OBSERVATION LABEL || SEQUENCE LABEL
                        var elementSDR = Element.Value; // ALL ELEMENT IN ONE SEQUENCE 

                        Console.WriteLine($"-------------- {observationClass} ---------------");

                        // CORTEX LAYER OUTPUT with elementSDR as INPUT and LEARN = TRUE
                        var lyrOut = layer1.Compute(elementSDR, learn);

                        // IF STABLE STATE ACHIEVED BREAK LOOP - 3
                        if (isInStableState)
                            break;

                    }

                }
                if (isInStableState)
                    break;
            }

            // ADDING TEMPORAL MEMEORY to CORTEX LAYER
            layer1.HtmModules.Add("tm", tm);
            
            string lastPredictedValue = "-1";
            List<string> lastPredictedValueList = new List<string>();
            double lastCycleAccuracy = 0;
            double accuracy = 0;

            List<List<string>> possibleSequence = new List<List<string>>();
            // TRAINING SP+TM TOGETHER
            foreach (var sequence in Sequences)  // SEQUENCE LOOP
            {
                int SequencesMatchCount = 0; // NUMBER OF MATCHES
                var tempLOGFILE = new Dictionary<int, string>();
                var tempLOGGRAPH = new Dictionary<int, double>();
                double SaturatedAccuracyCount = 0;


                for (int i = 0; i < maxCycles; i++) // MAXCYCLE LOOP 
                {
                    var ElementWisePrediction = new List<List<HtmClassifier<string,ComputeCycle>.ClassifierResult>>();
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

                        cls.Learn(observationLabel, actCells.ToArray());


                        // CLASS VOTING IS USED FOR SEQUENCE CLASSIFICATION EXPERIMENT i.e CANCER SEQUENCE CLASSIFICATION EXPERIMENT
                        if (!classVotingEnabled)
                        {

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
                        else {
                            if (lastPredictedValueList.Contains(observationLabel))
                            {
                                ElementMatches++;
                                lastPredictedValueList.Clear();
                                Debug.WriteLine($"Match. Actual value: {observationLabel} - Predicted value: {lastPredictedValue}");
                            }
                            else
                            {
                                Debug.WriteLine($"Mismatch! Actual value: {observationLabel} - Predicted values: {lastPredictedValue}");
                            }
                        }

                        Debug.WriteLine($"Col  SDR: {Helpers.StringifyVector(lyrOut.ActivColumnIndicies)}");
                        Debug.WriteLine($"Cell SDR: {Helpers.StringifyVector(actCells.Select(c => c.Index).ToArray())}");

                        if (learn == false)
                            Debug.WriteLine($"Inference mode");


                        if (lyrOut.PredictiveCells.Count > 0)
                        {
                            var predictedInputValue = cls.GetPredictedInputValues(lyrOut.PredictiveCells.ToArray(), 3);

                            Debug.WriteLine($"Current Input: {observationLabel}");
                            Debug.WriteLine("The predictions with similarity greater than 50% are");

                            foreach (var t in predictedInputValue)
                            {
                                
                                
                                if (t.Similarity >= (double)50.00)
                                {
                                    Debug.WriteLine($"Predicted Input: {string.Join(", ", t.PredictedInput)},\tSimilarity Percentage: {string.Join(", ", t.Similarity)}, \tNumber of Same Bits: {string.Join(", ", t.NumOfSameBits)}");
                                }

                                if (classVotingEnabled) {
                                    lastPredictedValueList.Add(t.PredictedInput);
                                }

                            }

                            if (!classVotingEnabled)
                            {
                                lastPredictedValue = predictedInputValue.First().PredictedInput;
                            }
                        }

                    }

                    accuracy = ((double)ElementMatches / (sequence.Count)) * 100;
                    Debug.WriteLine($"Cycle : {i} \t Accuracy:{accuracy}");
                    tempLOGGRAPH.Add(i, accuracy);
                    if (accuracy == 100) {
                        SequencesMatchCount++;
                        if (SequencesMatchCount >= 30)
                        {
                            tempLOGFILE.Add(i, $"Cycle : {i} \t  Accuracy:{accuracy} \t Number of times repeated {SequencesMatchCount}");
                            break;
                        }
                        tempLOGFILE.Add(i, $"Cycle : {i} \t  Accuracy:{accuracy} \t Number of times repeated {SequencesMatchCount}");

                    }

                    else if (lastCycleAccuracy == accuracy && accuracy != 0)
                    {
                        SaturatedAccuracyCount++;
                        if (SaturatedAccuracyCount >= 20 && lastCycleAccuracy > 70)
                        {
                            Debug.WriteLine($"NO FURTHER ACCURACY CAN BE ACHIEVED");
                            Debug.WriteLine($"Saturated Accuracy : {lastCycleAccuracy} \t Number of times repeated {SaturatedAccuracyCount}");
                            tempLOGFILE.Add(i, $"Cycle: { i} \t Accuracy:{accuracy} \t Number of times repeated {SaturatedAccuracyCount}");
                            break;
                        }
                        else {
                            tempLOGFILE.Add(i, $"Cycle: { i} \t Saturated Accuracy : {lastCycleAccuracy} \t Number of times repeated {SaturatedAccuracyCount}");
                        }
                    }
                    else
                    {
                        SaturatedAccuracyCount = 0;
                        SequencesMatchCount = 0;
                        lastCycleAccuracy = accuracy;
                        tempLOGFILE.Add(i, $"cycle : {i} \t Accuracy :{accuracy} \t ");
                    }
                    lastPredictedValueList.Clear();
                }
                tm.Reset(mem);
                learn = true;
                OUTPUT_LOG_LIST.Add(tempLOGFILE);
            }
           
            sw.Stop();

            //****************DISPLAY STATUS OF EXPERIMENT
            Debug.WriteLine("-------------------TRAINING END------------------------");
            Console.WriteLine("-----------------TRAINING END------------------------");
            Debug.WriteLine("-------------------WRTING TRAINING OUTPUT LOGS---------------------");
            Console.WriteLine("-------------------WRTING TRAINING OUTPUT LOGS------------------------");
            //*****************

            DateTime now = DateTime.Now;
            string filename = now.ToString("g");
            if (classVotingEnabled) 
            {
                filename = "CancerClassificationExperiment" + filename.Split(" ")[0]+"_" + now.Ticks.ToString() + ".txt";
            }
            else
            {
                filename = "PassengeCountPredictionExperiment" + filename.Split(" ")[0]+"_"+now.Ticks.ToString() + ".txt";
            }
            string path = System.AppDomain.CurrentDomain.BaseDirectory+"\\TrainingLogs\\" + filename;
            using (StreamWriter swOutput = File.CreateText(path))
            {
                swOutput.WriteLine($"{filename}");
                foreach (var SequencelogCycle in OUTPUT_LOG_LIST)
                {
                    swOutput.WriteLine("******Sequence Starting*****");
                    foreach (var cycleOutPutLog in SequencelogCycle)
                    {
                        swOutput.WriteLine(cycleOutPutLog.Value, true);
                    }
                    swOutput.WriteLine("****Sequence Ending*****");

                }
            }

            Debug.WriteLine("-------------------TRAINING LOGS HAS BEEN CREATED---------------------");
            Console.WriteLine("-------------------TRAINING LOGS HAS BEEN CREATED------------------------");

            var returnDictionary = new Dictionary<CortexLayer<object,object>, HtmClassifier<string,ComputeCycle>>();
            returnDictionary.Add(layer1, cls);
            return returnDictionary;
        }
        

    }
}