using System;
using System.IO;
using System.Linq;
using Microsoft.ML;
using Microsoft.ML.Data;
using CNTK;
using CNTKUtil;
using NeoCortexApi;
using NeoCortexApi.Entities;
using System.Collections.Generic;
using NeoCortexApi.Encoders;
using System.Diagnostics;
using NeoCortexApi.Network;
using NeoCortexApi.Classifiers;
using NeoCortex;
using NeoCortexApi.Utility;
using System.Xml;
using System.Xml.Serialization;
using LSTM_Comparison_Experiments;

namespace LSTM_Comparison_Experiments
{
    class ProgramExperiments_TaxiFarePrediction
    {
        static void Main(string[] args)
        {
            //Experiment1a.initiateExperiment();
            Experiment1b.initiateExperiment();
        }
    }
    public class TaxiTrip
    {

        [LoadColumn(1)] public DateTimeOffset tpep_pickup_datetime;
        [LoadColumn(3)] public int passenger_count;
        [LoadColumn(10)] public float FareAmount;

        public float[] GetFareAmount() => new float[] { FareAmount };
        public DateTimeOffset GetTimeofTheDay() => tpep_pickup_datetime;
        public int GetPassengerCount() => passenger_count;
        public int[] GetPassCntArray() => new int[] { passenger_count };
    }
    class Experiment1a
    {

        // Experiment 1-a 
        // details :
        // data : NYC TAXI 
        public static void initiateExperiment()
        {

            HelperMethods1a.run_HTM1a();
            HelperMethods1a.run_LSTM1a();
        }
    }
    class Experiment1b
    {
        // Experiment 1-b 
        // details :
        // data : NYC TAXI Half Hour Aggregated
        public static void initiateExperiment()
        {
            HelperMethods1b.run_HTM1b();
            //HelperMethods1b.run_LSTM1b();
        }
    }
    static class CommonHelperMethods
    {
        public static List<List<string>> fetchData(string path)
        {
            if (File.Exists(path))
            {

                List<string> listA = new List<string>();
                List<string> listB = new List<string>();
                using (var reader = new StreamReader(path))
                {
                    while (!reader.EndOfStream)
                    {
                        var line = reader.ReadLine();
                        var values = line.Split(',');

                        listA.Add(values[0]);
                        listB.Add(values[1]);
                    }
                }
                var x = new List<List<string>>();
                x.Add(listB);
                x.Add(listA);
                return x;
            }

            return null;

        }
        public static Parameters getParameters()
        {

            int inputBits = 2048;
            int numColumns = 2048;

            Parameters p = Parameters.getAllDefaultParameters();

            p.Set(KEY.RANDOM, new ThreadSafeRandom(42));
            p.Set(KEY.INPUT_DIMENSIONS, new int[] { inputBits });
            p.Set(KEY.COLUMN_DIMENSIONS, new int[] { numColumns });

            //p.Set(KEY.CELLS_PER_COLUMN, 25);
            //Model settings changed - ["Continuous online sequence Learning with an unsupervised neural network model"]
            p.Set(KEY.CELLS_PER_COLUMN, 32);

            p.Set(KEY.GLOBAL_INHIBITION, true);
            p.Set(KEY.LOCAL_AREA_DENSITY, -1); // In a case of global inhibition.

            //p.setNumActiveColumnsPerInhArea(10);
            // N of 40 (40= 0.02*2048 columns) active cells required to activate the segment.
            p.setNumActiveColumnsPerInhArea(0.02 * numColumns);
            // Activation threshold is 10 active cells of 40 cells in inhibition area.
            p.Set(KEY.POTENTIAL_RADIUS, 50);
            p.setInhibitionRadius(15);

            // Activates the high bumping/boosting of inactive columns.
            // This exeperiment uses HomeostaticPlasticityActivator, which will deactivate boosting and bumping.
            p.Set(KEY.MAX_BOOST, 10.0);
            p.Set(KEY.DUTY_CYCLE_PERIOD, 25);
            p.Set(KEY.MIN_PCT_OVERLAP_DUTY_CYCLES, 0.75);

            // Max number of synapses on the segment.
            p.setMaxNewSynapsesPerSegmentCount(32);

            // If learning process does not generate active segments, this value should be decreased. You can notice this with continious burtsing. look in trace for 'B.B.B'
            // If invalid patterns are predicted then this value should be increased.
            p.setActivationThreshold(15);
            p.setConnectedPermanence(0.5);

            // Learning is slower than forgetting in this case.
            p.setPermanenceDecrement(0.25);
            p.setPermanenceIncrement(0.15);

            // Used by punishing of segments.
            p.Set(KEY.PREDICTED_SEGMENT_DECREMENT, 0.01);

            return p;
        }
        public static Object getMLcontext(int choice)
        {
            var context = new MLContext();
            // set up the text loader 

            var textLoader = context.Data.CreateTextLoader(
                new TextLoader.Options()
                {
                    Separators = new[] { ',' },
                    HasHeader = true,
                    Columns = new[]
                    {
                        new TextLoader.Column("tpep_pickup_datetime", DataKind.DateTimeOffset, 1),
                        new TextLoader.Column("FareAmount", DataKind.Single, 10),
                        new TextLoader.Column("passenger_count", DataKind.Int32,3)
                    }
                }
            );
            if (choice == 1) { return textLoader; }
            else if (choice == 2) { return context; }
            else return null;

        }
        public static List<Object> fetchEncoders(float tempMin, float tempMax)
        {
            Dictionary<string, object> settingsScalarEncoder_PassCount = new Dictionary<string, object>()
            {
                { "W", 41},
                { "N", 1024},
                { "Radius", -1.0},
                { "MinVal", (double)tempMin},
                { "Periodic", false},
                { "Name", "scalar"},
                { "ClipInput", false},
                { "MaxVal", (double)tempMax}
            };
            ScalarEncoder encoder_PassCnt = new ScalarEncoder(settingsScalarEncoder_PassCount);
            ScalarEncoder encoder_DayOfWeek = new ScalarEncoder(new Dictionary<string, object>()
            {
                { "W", 3},
                { "N", 9},
                { "MinVal", (double)0}, // Min value = (0).
                { "MaxVal", (double)7}, // Max value = (7).
                { "Periodic", true}, // Since Monday would repeat again.
                { "Name", "Days Of Week"},
                { "ClipInput", true},
            });
            var now = DateTimeOffset.Now;
            Dictionary<string, Dictionary<string, object>> encoderSettingsDateTime = new Dictionary<string, Dictionary<string, object>>();

            encoderSettingsDateTime.Add("DateTimeEncoder", new Dictionary<string, object>()
                {
                    { "W", 21},
                    { "N", 1024},
                    { "MinVal", now.AddYears(-2)},
                    { "MaxVal", now},
                    { "Periodic", false},
                    { "Name", "DateTimeEncoder"},
                    { "ClipInput", false},
                    { "Padding", 5},
                });

            var encoderDateTime = new DateTimeEncoder(encoderSettingsDateTime, DateTimeEncoder.Precision.Days);
            List<Object> result = new List<object>();
            result.Add(encoder_PassCnt);
            result.Add(encoderDateTime);

            return result;

        }
        public static string GetKey(List<string> prevInputs, float input)
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

    }
    static class HelperMethods1a
    {
        static readonly string dataPath = Path.GetFullPath(@"C:\Users\Itachi_yv\Downloads\yellow_tripdata_2020-01.csv");
        static readonly string aggCount = Path.GetFullPath(@"C:\Users\Itachi_yv\Desktop\THESIS\Dataset\Experiment-1a\AggregatedCount.csv");
        static readonly string test_HTM_Model_Data_Experiment1a = Path.GetFullPath(@"C:\Users\Itachi_yv\Desktop\Test_HTM_Model.csv");
        public static List<float[][]> data_conversion_LSTM()
        {

            var processedData = CommonHelperMethods.fetchData(aggCount);

            var passCount = processedData.ElementAt(0);
            var dateTime = processedData.ElementAt(1);

            var dataToBeSent = new float[passCount.Count][];
            var passCntArray = new float[passCount.Count][];

            var list_to_be_Sent = new List<float[][]>();

            for (int i = 0; i < dateTime.Count; i++)
            {
                float passCnt_element = float.Parse(passCount.ElementAt(i));
                string dateTime_element = dateTime.ElementAt(i);
                var split_Datetime = dateTime_element.Substring(0, 10).Split("-");
                var dayOfWeek = ((float)Convert.ToDateTime(dateTime_element).DayOfWeek);



                float[] row = new float[] { dayOfWeek, float.Parse(split_Datetime.ElementAt(0)), float.Parse(split_Datetime.ElementAt(1)), float.Parse(split_Datetime.ElementAt(2)) };
                float[] passCntRow = new float[] { passCnt_element };
                dataToBeSent[i] = row;
                passCntArray[i] = passCntRow;
                Debug.WriteLine(dateTime_element);
            }

            list_to_be_Sent.Add(dataToBeSent);
            list_to_be_Sent.Add(passCntArray);

            return list_to_be_Sent;
        }
        public static void run_HTM1aqqq()
        {

            var dataFile = File.Exists(aggCount) ? aggCount : dataPath;
            var Processed_Data = CommonHelperMethods.fetchData(dataFile);


            var textLoader = (TextLoader)CommonHelperMethods.getMLcontext(1);
            var context = (MLContext)CommonHelperMethods.getMLcontext(2);

            // load the data 
            Console.Write("Loading training data....");
            var dataView = textLoader.Load(dataPath);
            Console.WriteLine("done");
            /*
            float tempMax = 0; float tempMin = 0;
            foreach (var inputRow in (Processed_Data.ElementAt(0)))
            {
                var currentRow = float.Parse(inputRow);
                if (currentRow > tempMax) tempMax = currentRow;
                if (currentRow < tempMin) tempMin = currentRow;
            }
            List<Object> encoders_Collection = CommonHelperMethods.fetchEncoders(tempMin, tempMax);
            var DateTimeEncoder = (DateTimeEncoder)encoders_Collection[0];
            var ScalarEncoder = (ScalarEncoder)encoders_Collection[1];
            */

            /*
             * 1. Time Of Day
             * 2. Week Of Day
             * 3. Date
             * 4. Pass Count
             */

            int inputBits = 2048;
            int numColumns = 2048;

            NeoCortexApi.Entities.Parameters p = CommonHelperMethods.getParameters();
            List<int[]> encodedDataSet = new List<int[]>();
            List<float> inputValue = new List<float>();

            var PassCount = Processed_Data.ElementAt(0);
            var dateTime = Processed_Data.ElementAt(1);

            for (var i = 0; i < dateTime.Count; i++)
            {
                var dateTimeInternal = DateTime.Parse(dateTime[i]);
                var passCountInternal = PassCount.ElementAt(i);
                int[] encoded_PassCnt = new int[1024];
                if (dateTimeInternal.Year == 2020)
                {
                    // encoded_PassCnt = ScalarEncoder.Encode((int)dateTimeInternal.DayOfWeek);
                    DateTimeOffset time2 = new DateTimeOffset(dateTimeInternal, TimeZoneInfo.FindSystemTimeZoneById("Central Standard Time").GetUtcOffset(dateTimeInternal));
                    //     int[] encoded_DateTime = DateTimeEncoder.Encode(time2);
                    //         encoded_PassCnt = ScalarEncoder.Encode(passCountInternal);
                    //           int[] combine_EncodedValues = new int[encoded_PassCnt.Length + encoded_DateTime.Length];
                    // /            Array.Copy(encoded_PassCnt, combine_EncodedValues, encoded_PassCnt.Length);
                    //           Array.Copy(encoded_DateTime, 0, combine_EncodedValues, encoded_PassCnt.Length, encoded_DateTime.Length);
                    //inputValue.Add(date.Value);
                    //       encodedDataSet.Add(combine_EncodedValues);
                    inputValue.Add(float.Parse(passCountInternal));
                }
            }

            //     RunExperiment1a(inputBits, p, ScalarEncoder, encodedDataSet, inputValue);

        }
        public static void run_HTM1a()
        {

            var dataFile = File.Exists(aggCount) ? aggCount : dataPath;
            var Processed_Data = CommonHelperMethods.fetchData(dataFile);


            var textLoader = (TextLoader)CommonHelperMethods.getMLcontext(1);
            var context = (MLContext)CommonHelperMethods.getMLcontext(2);

            // load the data 
            Console.Write("Loading training data....");
            var dataView = textLoader.Load(dataPath);
            Console.WriteLine("done");

            float tempMax = 0; float tempMin = 0;
            foreach (var inputRow in (Processed_Data.ElementAt(0)))
            {
                var currentRow = float.Parse(inputRow);
                if (currentRow > tempMax) tempMax = currentRow;
                if (currentRow < tempMin) tempMin = currentRow;
            }
            List<Object> encoders_Collection = CommonHelperMethods.fetchEncoders(tempMin, tempMax);
            var DateTimeEncoder = (DateTimeEncoder)encoders_Collection[1];
            var ScalarEncoder = (ScalarEncoder)encoders_Collection[0];

            int inputBits = 2048;
            int numColumns = 2048;

            NeoCortexApi.Entities.Parameters p = CommonHelperMethods.getParameters();
            List<int[]> encodedDataSet = new List<int[]>();
            List<float> inputValue = new List<float>();

            var PassCount = Processed_Data.ElementAt(0);
            var dateTime = Processed_Data.ElementAt(1);

            for (var i = 0; i < dateTime.Count; i++)
            {
                var dateTimeInternal = DateTime.Parse(dateTime[i]);
                var passCountInternal = PassCount.ElementAt(i);
                int[] encoded_PassCnt = new int[1024];
                if (dateTimeInternal.Year == 2020)
                {
                    // encoded_PassCnt = ScalarEncoder.Encode((int)dateTimeInternal.DayOfWeek);
                    DateTimeOffset time2 = new DateTimeOffset(dateTimeInternal, TimeZoneInfo.FindSystemTimeZoneById("Central Standard Time").GetUtcOffset(dateTimeInternal));
                    int[] encoded_DateTime = DateTimeEncoder.Encode(time2);
                    encoded_PassCnt = ScalarEncoder.Encode(passCountInternal);
                    int[] combine_EncodedValues = new int[encoded_PassCnt.Length + encoded_DateTime.Length];
                    Array.Copy(encoded_PassCnt, combine_EncodedValues, encoded_PassCnt.Length);
                    Array.Copy(encoded_DateTime, 0, combine_EncodedValues, encoded_PassCnt.Length, encoded_DateTime.Length);
                    //inputValue.Add(date.Value);
                    encodedDataSet.Add(combine_EncodedValues);
                    inputValue.Add(float.Parse(passCountInternal));
                }
            }

            RunExperiment1a(inputBits, p, ScalarEncoder, encodedDataSet, inputValue);

        }
        public static void run_LSTM1a()
        {
            var dataFile = File.Exists(aggCount) ? aggCount : dataPath;
            var Processed_Data = CommonHelperMethods.fetchData(dataFile);
            var textLoader = (TextLoader)CommonHelperMethods.getMLcontext(1);
            var context = (MLContext)CommonHelperMethods.getMLcontext(2);

            // load the data 
            Console.Write("Loading training data....");
            var dataView = textLoader.Load(dataPath);
            Console.WriteLine("done");

            // load training data
            var training = context.Data.CreateEnumerable<TaxiTrip>(dataView, reuseRowObject: false);

            var features = NetUtil.Var(new int[] { 2 }, DataType.Float);
            var labels = NetUtil.Var(new int[] { 1 }, DataType.Float);

            // build a regression model
            var network = features
                .Dense(1)
                .ToNetwork();
            Console.WriteLine("Model architecture:");
            Console.WriteLine(network.ToSummary());

            // set up the loss function and the classification error function
            var lossFunc = NetUtil.MeanSquaredError(network.Output, labels);
            var errorFunc = NetUtil.MeanAbsoluteError(network.Output, labels);

            // set up a trainer
            var learner = network.GetAdamLearner(
                learningRateSchedule: (0.001, 1),
                momentumSchedule: (0.9, 1),
                unitGain: false);

            // set up a trainer and an evaluator
            var trainer = network.GetTrainer(learner, lossFunc, errorFunc);

            // train the model
            Console.WriteLine("Epoch\tTrain\tTrain");
            Console.WriteLine("\tLoss\tError");
            Console.WriteLine("-----------------------");
            //int minibatchSize = 64;
            //int numMinibatchesToTrain = 1000;

            var maxEpochs = 512; // 50;
            var batchSize = 32; // 32;
            var loss = new double[maxEpochs];
            var trainingError = new double[maxEpochs];
            var batchCount = 0;
            var fetchDataBatch = data_conversion_LSTM();
            for (int epoch = 0; epoch < maxEpochs; epoch++)
            {
                // train one epoch on batches
                loss[epoch] = 0.0;
                trainingError[epoch] = 0.0;
                batchCount = 0;
                fetchDataBatch.ElementAt(0).Index().Shuffle().Batch(batchSize, (indices, begin, end) =>
                {
                    // get the current batch
                    var featureBatch = features.GetBatch(fetchDataBatch.ElementAt(0), indices, begin, end);
                    var labelBatch = labels.GetBatch(fetchDataBatch.ElementAt(1), indices, begin, end);

                    // train the regression model on the batch
                    var result = trainer.TrainBatch(
                        new[] {
                            (features, featureBatch),
                            (labels,  labelBatch)
                        },
                        false
                    );
                    var x = trainer.TestBatch(
                        new[]{(features, featureBatch),
                            (labels,  labelBatch)
                        });
                    loss[epoch] += result.Loss;
                    trainingError[epoch] += result.Evaluation;
                    batchCount++;
                });
                // show results
                loss[epoch] /= batchCount;
                trainingError[epoch] /= batchCount;
                Console.WriteLine($"{epoch}\t{loss[epoch]:F3}\t{trainingError[epoch]:F3}");
            }
            // show final results
            var finalError = trainingError[maxEpochs - 1];
            Console.WriteLine();
            Console.WriteLine($"Final MAE: {finalError:0.00}");
        }
        public static void RunExperiment1a(int inputBits, Parameters p, EncoderBase encoder, List<int[]> inputValues, List<float> inputValuesString)
        {
            Stopwatch sw = new Stopwatch();
            sw.Start();

            int maxMatchCnt = 0;
            bool learn = true;

            CortexNetwork net = new CortexNetwork("my cortex");
            List<CortexRegion> regions = new List<CortexRegion>();
            CortexRegion region0 = new CortexRegion("1st Region");

            regions.Add(region0);

            var mem = new Connections();

            p.apply(mem);

            bool isInStableState;

            //HtmClassifier<double, ComputeCycle> cls = new HtmClassifier<double, ComputeCycle>();
            HtmClassifier<string, ComputeCycle> cls = new HtmClassifier<string, ComputeCycle>();
            var numInputs = 44;
            TemporalMemory tm1 = new TemporalMemory();
            HomeostaticPlasticityController hpa = new HomeostaticPlasticityController(mem, 44, (isStable, numPatterns, actColAvg, seenInputs) =>
            {
                if (isStable)
                    // Event should be fired when entering the stable state.
                    Debug.WriteLine($"STABLE: Patterns: {numPatterns}, Inputs: {seenInputs}, iteration: {seenInputs / numPatterns}");
                else
                    // Ideal SP should never enter unstable state after stable state.
                    Debug.WriteLine($"INSTABLE: Patterns: {numPatterns}, Inputs: {seenInputs}, iteration: {seenInputs / numPatterns}");


                if (numPatterns != numInputs)
                    throw new InvalidOperationException("Stable state must observe all input patterns");
                isInStableState = true;
                cls.ClearState();
                tm1.Reset(mem);
                //  }, numOfCyclesToWaitOnChange: 25); // Configuration -0
                //}, numOfCyclesToWaitOnChange: 35); // Configuration -1
                //}, numOfCyclesToWaitOnChange: 25); // Configuration -1
            }, numOfCyclesToWaitOnChange: 30); // Configuration -1
            SpatialPoolerMT sp1 = new SpatialPoolerMT(hpa);
            sp1.Init(mem, new DistributedMemory()
            {
                ColumnDictionary = new InMemoryDistributedDictionary<int, NeoCortexApi.Entities.Column>(1),
            });

            tm1.Init(mem);

            CortexLayer<object, object> layer1 = new CortexLayer<object, object>("L1");
            region0.AddLayer(layer1);

            //    layer1.HtmModules.Add("encoder", encoder);
            layer1.HtmModules.Add("sp", sp1);
            layer1.HtmModules.Add("tm", tm1);

            //double[] inputs = inputValues.ToArray();
            Dictionary<double, int[]> prevActiveCols = new Dictionary<double, int[]>();
            //int[] prevActiveCols = new int[0];
            int cycle = 0;
            int matches = 0;
            string lastPredictedValue = "";
            String prediction = null;

            Dictionary<float, List<List<int>>> activeColumnsLst = new Dictionary<float, List<List<int>>>();
            //foreach (var input in inputValuesString)
            for (int i = 0; i < inputValuesString.Count; i++)
            {
                //  var lyrOut = layer1.Compute((object)inputValues[i], learn) as ComputeCycle;
                if (activeColumnsLst.ContainsKey(inputValuesString[i]) == false)
                    activeColumnsLst.Add(inputValuesString[i], new List<List<int>>());
            }
            int maxCycles = 500;
            int maxPrevInputs = inputValues.Count - 1;
            //int maxPrevInputs = 20;
            List<string> previousInputs = new List<string>();
            previousInputs.Add("");

            Debug.WriteLine(p.ToString());

            // Now training with SP+TM. SP is pretrained on the given input pattern.
            for (int i = 0; i < maxCycles; i++)
            {
                matches = 0;

                cycle++;

                Debug.WriteLine($"-------------- Cycle {cycle} ---------------");
                var j = 0;
                //for (int indx = 0; indx < 21; indx++)
                foreach (var input in inputValuesString)
                {
                    //  var input = input_FareAmount[indx];
                    Debug.WriteLine($"-------------- {input} ---------------");

                    var lyrOut = layer1.Compute(inputValues[j], learn) as ComputeCycle;
                    j++; if (j == inputValuesString.Count) { j = 0; }

                    var activeColumns = layer1.GetResult("sp") as int[];

                    activeColumnsLst[input].Add(activeColumns.ToList());

                    previousInputs.Add(input.ToString());
                    if (previousInputs.Count > (maxPrevInputs + 1))
                        previousInputs.RemoveAt(0);

                    string key = CommonHelperMethods.GetKey(previousInputs, input);

                    cls.Learn(key, lyrOut.ActiveCells.ToArray());


                    if (learn == false)
                        Debug.WriteLine($"Inference mode");

                    Debug.WriteLine($"Col  SDR: {Helpers.StringifyVector(lyrOut.ActivColumnIndicies)}");
                    Debug.WriteLine($"Cell SDR: {Helpers.StringifyVector(lyrOut.ActiveCells.Select(c => c.Index).ToArray())}");

                    if (key == lastPredictedValue)
                    {
                        matches++;
                        Debug.WriteLine($"Match. Actual value: {key} - Predicted value: {lastPredictedValue}");
                    }
                    else
                        Debug.WriteLine($"Missmatch! Actual value: {key} - Predicted value: {lastPredictedValue}");

                    if (lyrOut.PredictiveCells.Count > 0)
                    {
                        var predictedInputValue = cls.GetPredictedInputValue(lyrOut.PredictiveCells.ToArray());

                        Debug.WriteLine($"Current Input: {input}");
                        Debug.WriteLine("The predictions with similarity greater than 50% are");
                        //Debug.WriteLine($"Predicted Input: {string.Join(", ", predictedInputValue)},\tSimilarity Percentage: {string.Join(", ", t.Similarity)}, \tNumber of Same Bits: {string.Join(", ", t.NumOfSameBits)}");


                        if (predictedInputValue != null)
                        {
                            lastPredictedValue = predictedInputValue;
                        }
                        else
                        {
                            lastPredictedValue = "";
                        }


                    }
                    else
                    {
                        Debug.WriteLine($"NO CELLS PREDICTED for next cycle.");
                        lastPredictedValue = String.Empty;
                    }
                }

                double accuracy = (double)matches / (double)maxPrevInputs * 100.0;
                Debug.WriteLine($"Cycle: {cycle}\tMatches={matches} of {maxPrevInputs}\t {accuracy}%");
                //  Debug.WriteLine($"Cycle: {cycle}\tMatches={matches} of {inputValues.Count}\t {accuracy}%");

                if (accuracy >= 100.0)
                {
                    maxMatchCnt++;
                    Debug.WriteLine($"100% accuracy reched {maxMatchCnt} times.");
                    if (maxMatchCnt >= 30)
                    {
                        sw.Stop();
                        Debug.WriteLine($"Exit experiment in the stable state after 30 repeats with 100% of accuracy. Elapsed time: {sw.ElapsedMilliseconds / 1000 / 60} min.");
                        learn = false;

                        //
                        // This code snippet starts with some input value and tries to predict all next inputs
                        // as they have been learned as a sequence.
                        // We take a random value to start somwhere in the sequence.

                        /*
                         * Testing Model
                         */

                        test_HTM_model(inputValues, layer1, cls, inputValuesString);
                        break;
                    }
                    else if (maxMatchCnt > 0)
                    {
                        Debug.WriteLine($"At 100% accuracy after {maxMatchCnt} repeats we get a drop of accuracy with {accuracy}. This indicates instable state. Learning will be continued.");
                        //      if(accuracy<100) maxMatchCnt = 0;
                        if (j == inputValuesString.Count) { matches = 0; }

                    }
                }

                Debug.WriteLine("---- cell state trace ----");

                /*                    cls.TraceState($"cellState_MinPctOverlDuty-{p[KEY.MIN_PCT_OVERLAP_DUTY_CYCLES]}_MaxBoost-{p[KEY.MAX_BOOST]}.csv");
                */
                Debug.WriteLine("---- Spatial Pooler column state  ----");
                Debug.WriteLine("------------ END ------------");

            }
        }
        public static void test_HTM_model(List<int[]> inputValues, CortexLayer<object, object> layer1, HtmClassifier<string, ComputeCycle> classifier, List<float> inputValueString)
        {
            var predictedValues = new List<double>();
            var predictedValuesString = new List<String>();
            List<int[]> testing_datset = new List<int[]>();

            double dataLength = Convert.ToDouble(inputValues.Count) * 0.7;

            for (int i = 0; i < dataLength; i++)
            {
                Random r = new Random();
                var randomIndex = r.Next(0, (int)dataLength);
                var datarow = inputValues[randomIndex];
                var lyrOut = layer1.Compute(datarow, false) as ComputeCycle;
                var getPredictedValue = classifier.GetPredictedInputValues(lyrOut.PredictiveCells.ToArray(), 1);
                //var getPredictedValue = classifier.GetPredictedInputValue(lyrOut.PredictiveCells.ToArray());
                //var listPrediction = classifier.GetPredictedInputValues(lyrOut.PredictiveCells.ToArray(),10).Select(x=>x.Similarity>=80);
                //foreach (var item in getPredictedValue)
                Debug.WriteLine($"Input -> {inputValueString.ElementAt(randomIndex)} Predicted Value -> {getPredictedValue[0].PredictedInput} Similarity -> {getPredictedValue[0].PredictedInput}");
                //predictedValues.Add(getPredictedValue);
                predictedValuesString.Add(getPredictedValue[0].Similarity + " , " + getPredictedValue[0].PredictedInput);
            }



            using (StreamWriter w = File.AppendText(test_HTM_Model_Data_Experiment1a))
            {
                int k = 0;
                foreach (var line in predictedValuesString)
                {
                    w.WriteLine(line);
                    //   w.WriteLine(predictedValuesString.ElementAt(k));
                    // k++;

                }
            }



        }
    }
    static class HelperMethods1b
    {
        static readonly string aggCount = Path.GetFullPath(@"C:\Users\Itachi_yv\Desktop\Aggregated30Count.csv");
        static readonly string aggCount2 = Path.GetFullPath(@"C:\Users\Itachi_yv\Desktop\Aggregated30Count2.csv");
        static readonly string test_HTM_Model_Data_Experiment1a = Path.GetFullPath(@"C:\Users\Itachi_yv\Desktop\Test_HTM_Model.csv");
        static readonly string test_HTM_Similarity_Experiment1a = Path.GetFullPath(@"C:\Users\Itachi_yv\Desktop\Test_HTM_Model_PredictionSIM.csv");
        public static void HourAggregatedData()
        {
            var data = HelperMethods1b.fetchData(aggCount);

            var passCnt = data.ElementAt(0);
            var dateTime = data.ElementAt(1);
            List<DateTime> allDateTimes = new List<DateTime>();
            List<float> allPassCnts = new List<float>();


            for (int i = 0; i < passCnt.Count; i++)
            {
                var convertedDate = Convert.ToDateTime(dateTime.ElementAt(i));
                allDateTimes.Add(convertedDate);
                var floatPassCnt = 0f;
                if (passCnt.ElementAt(i).Equals(""))
                {
                    floatPassCnt = 0f;
                }
                else
                {
                    floatPassCnt = float.Parse(passCnt.ElementAt(i));
                }

                allPassCnts.Add(floatPassCnt);
            }

            var y = allDateTimes.GroupBy(c => c.Date).ToArray();

            foreach (var date_element in y)
            {
                var date_element_key = date_element.Key;
            }
            Debug.WriteLine(y);

        }
        public static void getHourAgg(System.Linq.IGrouping<System.DateTime, System.DateTime>[] data)
        {
            var date_collector = new Dictionary<System.DateTime, int[]>();

            foreach (var day in data)
            {
                date_collector.Add(day.Key, new int[48]);
                Debug.WriteLine(day);
                foreach (var eachEntry in day)
                {
                    var timeOfDay = eachEntry.TimeOfDay;
                    var hour = timeOfDay.Hours;
                    var minutes = timeOfDay.Minutes;
                    var no_of_partition = hour * 2;
                    if (minutes > 30) { no_of_partition++; }
                    var count_present = date_collector[day.Key];
                    count_present[no_of_partition] = count_present[no_of_partition] + 1;
                    date_collector[day.Key] = count_present;

                }

            }
            using (StreamWriter w = File.AppendText(aggCount))
            {
                int k = 0;
                foreach (var line in date_collector)
                {
                    var date = line.Key.Date;
                    var values_for_Day = line.Value;
                    w.WriteLine(date.ToString() + "-" + String.Join('-', values_for_Day));
                }
            }

        }
        public static List<List<string>> fetchData(string path)
        {
            List<string> listA = new List<string>();
            List<string> listB = new List<string>();

            if (File.Exists(path))
            {
                int lines = 0;
                using (var reader = new StreamReader(path))
                {
                    while (!reader.EndOfStream)
                    {
                        var line = reader.ReadLine();
                        var values = line.Split(',');

                        listA.Add(values[0]);
                        listB.Add(values[1]);
                        lines++;
                    }
                }
                var x = new List<List<string>>();
                x.Add(listB);
                x.Add(listA);
                return x;
            }
            else
            {
                return null;
            }


        }
        public static void run_HTM1b()
        {

            var Processed_Data = fetchData(aggCount2);

            var textLoader = (TextLoader)CommonHelperMethods.getMLcontext(1);
            var context = (MLContext)CommonHelperMethods.getMLcontext(2);

            // load the data 
            Console.Write("Loading training data....");
            //var dataView = textLoader.Load(aggCount);
            Console.WriteLine("done");
            float tempMax = 0; float tempMin = 0;

            foreach (var inputRow in (Processed_Data.ElementAt(0)))
            {
                var currentRow = float.Parse(inputRow);
                if (currentRow > tempMax) tempMax = currentRow;
                if (currentRow < tempMin) tempMin = currentRow;
            }

            List<Object> encoders_Collection = CommonHelperMethods.fetchEncoders(tempMin, tempMax);

            var DateTimeEncoder = (DateTimeEncoder)encoders_Collection[1];
            var ScalarEncoder = (ScalarEncoder)encoders_Collection[0];

            int inputBits = 2048;
            int numColumns = 2048;

            NeoCortexApi.Entities.Parameters p = CommonHelperMethods.getParameters();
            List<int[]> encodedDataSet = new List<int[]>();
            List<string> inputValue = new List<string>();

            var PassCount = Processed_Data.ElementAt(0);
            var dateTime = Processed_Data.ElementAt(1);
            
            for (var i = 0; i < dateTime.Count; i++)
            {
                var dateTimeInternal = DateTimeOffset.Parse(dateTime[i]);
                var passCountInternal = PassCount.ElementAt(i);
                if (dateTimeInternal.Date.Year == 2020 && dateTimeInternal.Date.Month==1)
                {
                    var encoded_PassCnt = ScalarEncoder.Encode(passCountInternal);
                    var encodedDateTime = DateTimeEncoder.Encode(dateTimeInternal);
                    var encoded_combined = new int[2048];
                    Array.Copy(encodedDateTime, encoded_combined, encodedDateTime.Length);
                    Array.Copy(encoded_PassCnt, 0, encoded_combined, 1024, encoded_PassCnt.Length);
                    encodedDataSet.Add(encoded_combined);
                    inputValue.Add(passCountInternal.ToString());
                }

            }
            RunExperiment1b(inputBits, p, ScalarEncoder, encodedDataSet, inputValue);
        }
        public static void RunExperiment1b(int inputBits, Parameters p, EncoderBase encoder, List<int[]> inputValues, List<string> inputValuesString)
        {
            Stopwatch sw = new Stopwatch();
            sw.Start();

            int maxMatchCnt = 0;
            bool learn = true;

            CortexNetwork net = new CortexNetwork("my cortex");
            List<CortexRegion> regions = new List<CortexRegion>();
            CortexRegion region0 = new CortexRegion("1st Region");

            regions.Add(region0);

            var mem = new Connections();

            p.apply(mem);

            bool isInStableState;

            //HtmClassifier<double, ComputeCycle> cls = new HtmClassifier<double, ComputeCycle>();
            HtmClassifier<string, ComputeCycle> cls = new HtmClassifier<string, ComputeCycle>();
            var numInputs = inputValues.Count;
            TemporalMemory tm1 = new TemporalMemory();
            HomeostaticPlasticityController hpa = new HomeostaticPlasticityController(mem, inputValues.Count, (isStable, numPatterns, actColAvg, seenInputs) =>
            {
                if (isStable)
                    // Event should be fired when entering the stable state.
                    Debug.WriteLine($"STABLE: Patterns: {numPatterns}, Inputs: {seenInputs}, iteration: {seenInputs / numPatterns}");
                else
                    // Ideal SP should never enter unstable state after stable state.
                    Debug.WriteLine($"INSTABLE: Patterns: {numPatterns}, Inputs: {seenInputs}, iteration: {seenInputs / numPatterns}");


                if (numPatterns != numInputs)
                    throw new InvalidOperationException("Stable state must observe all input patterns");
                isInStableState = true;
                cls.ClearState();
                tm1.Reset(mem);
                //  }, numOfCyclesToWaitOnChange: 25); // Configuration -0
                //}, numOfCyclesToWaitOnChange: 35); // Configuration -1
                //}, numOfCyclesToWaitOnChange: 25); // Configuration -1
            }, numOfCyclesToWaitOnChange: 50); // Configuration -1
            SpatialPoolerMT sp1 = new SpatialPoolerMT(hpa);
            sp1.Init(mem, new DistributedMemory()
            {
                ColumnDictionary = new InMemoryDistributedDictionary<int, NeoCortexApi.Entities.Column>(1),
            });

            tm1.Init(mem);

            CortexLayer<object, object> layer1 = new CortexLayer<object, object>("L1");
            region0.AddLayer(layer1);

            //    layer1.HtmModules.Add("encoder", encoder);
            layer1.HtmModules.Add("sp", sp1);
            layer1.HtmModules.Add("tm", tm1);

            //double[] inputs = inputValues.ToArray();
            Dictionary<double, int[]> prevActiveCols = new Dictionary<double, int[]>();
            //int[] prevActiveCols = new int[0];
            int cycle = 0;
            int matches = 0;
            string lastPredictedValue = "";
            String prediction = null;

            Dictionary<float, List<List<int>>> activeColumnsLst = new Dictionary<float, List<List<int>>>();
            //foreach (var input in inputValuesString)
            for (int i = 0; i < inputValuesString.Count; i++)
            {
                
                var passCnt_temp = float.Parse(inputValuesString[i]);
                //  var lyrOut = layer1.Compute((object)inputValues[i], learn) as ComputeCycle;
                if (activeColumnsLst.ContainsKey(passCnt_temp) == false)
                    activeColumnsLst.Add(passCnt_temp, new List<List<int>>());
            }
            int maxCycles = 500;
            int maxPrevInputs = inputValues.Count - 1;
            //int maxPrevInputs = 20;
            List<string> previousInputs = new List<string>();
            previousInputs.Add("");

            Debug.WriteLine(p.ToString());

            // Now training with SP+TM. SP is pretrained on the given input pattern.
            for (int i = 0; i < maxCycles; i++)
            {
                matches = 0;
                cycle++;
                Debug.WriteLine($"-------------- Cycle {cycle} ---------------");
                for (int i1 = 0; i1 < inputValues.Count; i1++) 
                {

                        float labelValue = float.Parse(inputValuesString.ElementAt(i1));
                        var encoded_features = inputValues.ElementAt(i1);
                        
                        Debug.WriteLine($"-------------- {inputValuesString.ElementAt(i1)} ---------------");

                        var lyrOut = layer1.Compute(encoded_features, learn) as ComputeCycle;

                        var activeColumns = layer1.GetResult("sp") as int[];

                        activeColumnsLst[labelValue].Add(activeColumns.ToList());

                        previousInputs.Add(inputValuesString.ElementAt(i1));
                        if (previousInputs.Count > (maxPrevInputs + 1))
                            previousInputs.RemoveAt(0);

                        string key = CommonHelperMethods.GetKey(previousInputs, labelValue);

                        //cls.Learn(GetKey(prevInput, input), lyrOut.ActiveCells.ToArray());
                        cls.Learn(key, lyrOut.ActiveCells.ToArray());

                        if (learn == false)
                            Debug.WriteLine($"Inference mode");

                        //Debug.WriteLine($"Col  SDR: {Helpers.StringifyVector(lyrOut.ActivColumnIndicies)}");
                        //Debug.WriteLine($"Cell SDR: {Helpers.StringifyVector(lyrOut.ActiveCells.Select(c => c.Index).ToArray())}");

                        if (key == lastPredictedValue)
                        {
                            matches++;
                            Debug.WriteLine($"Match. Actual value: {key} - Predicted value: {lastPredictedValue}");
                        }
                        else
                            Debug.WriteLine($"Missmatch! Actual value: {key} - Predicted value: {lastPredictedValue}");

                        if (lyrOut.PredictiveCells.Count > 0)
                        {
                            var predictedInputValue = cls.GetPredictedInputValue(lyrOut.PredictiveCells.ToArray());

                            Debug.WriteLine($"Current Input: {inputValuesString.ElementAt(i1)} \t| Predicted Input: {predictedInputValue}");

                            lastPredictedValue = predictedInputValue;
                        }
                        else
                        {
                            Debug.WriteLine($"NO CELLS PREDICTED for next cycle.");
                            lastPredictedValue = String.Empty;
                        }

                }

                    // The brain does not do that this way, so we don't use it.
                    // tm1.reset(mem);

                    double accuracy = (double)matches / (double)inputValues.Count * 100.0;

                    Debug.WriteLine($"Cycle: {cycle}\tMatches={matches} of {inputValues.Count}\t {accuracy}%");

                    if (accuracy == 100.0)
                    {
                        maxMatchCnt++;
                        Debug.WriteLine($"100% accuracy reched {maxMatchCnt} times.");
                        if (maxMatchCnt >= 30)
                        {
                            sw.Stop();
                            Debug.WriteLine($"Exit experiment in the stable state after 30 repeats with 100% of accuracy. Elapsed time: {sw.ElapsedMilliseconds / 1000 / 60} min.");
                            learn = false;
                            break;
                        }
                    }
                    else if (maxMatchCnt > 0)
                    {
                        Debug.WriteLine($"At 100% accuracy after {maxMatchCnt} repeats we get a drop of accuracy with {accuracy}. This indicates instable state. Learning will be continued.");
                        maxMatchCnt = 0;
                    }
                
            }
        }
        public static void run_LSTM1b() { }

    } 
}
