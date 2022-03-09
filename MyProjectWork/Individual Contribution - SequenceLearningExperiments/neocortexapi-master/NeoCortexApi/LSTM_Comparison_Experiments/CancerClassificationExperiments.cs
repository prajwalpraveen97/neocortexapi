using System;


// Experiment 2 
// details :
// data : CANCER CLASSIFICATION


using CNTKUtil;
using Microsoft.ML;
using Microsoft.ML.Data;
using NeoCortexApi;
using NeoCortexApi.Classifiers;
using NeoCortexApi.Encoders;
using NeoCortexApi.Entities;
using NeoCortexApi.Network;
using System;
using System.Collections.Generic;
using System.IO;
using System.Text;
using System.Linq;
using System.Diagnostics;
using LSTM_Comparison_Experiments;

public class Experiment2a
{
    
    static void Main(string[] args)
    {
        LungCancerHelperMethod_Exp1.run_HTM_LC1();
    } 
    
        public static class LungCancerHelperMethod_Exp1
        {
            static readonly string dataPath = Path.GetFullPath(@"C:\Users\Itachi_yv\Desktop\ACPs_Lung_cancer.csv");
            static readonly string[] classes = new string[] { "mod. active", "very active", "inactive - exp", "inactive - virtual" };
            public static Dictionary<string, List<string>> fetchData(string path, int skipRows)
            {
                var lines = 0;
                if (File.Exists(path))
                {
                    var data_dict = new Dictionary<string, List<string>>();
                    data_dict.Add(classes[0], new List<string>());
                    data_dict.Add(classes[1], new List<string>());
                    data_dict.Add(classes[2], new List<string>());
                    data_dict.Add(classes[3], new List<string>());

                    using (var reader = new StreamReader(path))
                    {
                        while (!reader.EndOfStream)
                        {
                            if (lines > skipRows)
                            {
                                var line = reader.ReadLine();
                                var values = line.Split(',');
                                var list_from_dict = data_dict[values[2]];
                                list_from_dict.Add(values[1]);
                                data_dict[values[2]] = list_from_dict;
                            }
                            else
                            {
                                var line = reader.ReadLine();
                            }
                            lines++;
                        }
                    }
                    return data_dict;
                }

                return null;

            }
            public static void run_HTM_LC1()
            {

                var dataFile = dataPath;
                var Processed_Data = fetchData(dataFile, 1);


                var textLoader = (TextLoader)CommonHelperMethods.getMLcontext(1);
                var context = (MLContext)CommonHelperMethods.getMLcontext(2);

                // load the data 
                Console.Write("Loading training data....");
                var dataView = textLoader.Load(dataPath);
                Console.WriteLine("done");

                List<Object> encoders_Collection = fetchEncoders(Processed_Data.Keys.ToArray());
                var Category_Encoder = (CategoryEncoder)encoders_Collection[1];
                var ScalarEncoder = (ScalarEncoder)encoders_Collection[0];
                int inputBits = 570;
                NeoCortexApi.Entities.Parameters p = CommonHelperMethods.getParameters();
                List<int[]> encodedDataSet = new List<int[]>();
                List<List<string>> data = new List<List<string>>();

                Dictionary<string, List<int[]>> encoded_Dataset = new Dictionary<string, List<int[]>>();
                var k = 0;
                foreach (var key in Processed_Data.Keys)
                {
                    var list_values = Processed_Data[key];
                    var category = key+k;
                        
                    list_values.Sort(delegate (string x, string y) {
                        return x.Length.CompareTo(y.Length);
                    });
                    List<int[]> encoded_collected = new List<int[]>();
                    foreach (var value in list_values)
                    {
                        List<int[]> encoded_individual = new List<int[]>();
                        foreach (var ch in value)
                        {
                            encoded_individual.Add(ScalarEncoder.Encode((int)char.ToUpper(ch) - 64));
                        }
                        int[] result = new int[0];
                        foreach (var x in encoded_individual)
                        {
                            result = result.Concat(x).ToArray();
                        }
                        encoded_collected.Add(result);

                    }
                k++;
                    encoded_Dataset.Add(key, encoded_collected);
                }
                RunExperiment_LC1(inputBits, p, ScalarEncoder, encoded_Dataset, Processed_Data);
            }

            public static int[] length_padding(int[] data, int length)
            {

                int length_diff = length - data.Length;
                int[] padd_arr = new int[length_diff];
                data = data.Concat(padd_arr).ToArray();
                return data;
            }
            public static List<string> encode_string_processes(string value)
            {
                List<string> encoded = new List<string>();
                List<string> post_encoded = new List<string>();
                for (int i = 0; i < value.Length; i++)
                {
                    if (i + 3 > value.Length)
                    {
                        string sub_str1 = value.Substring(i, value.Length - i);
                        string num = "";
                        foreach (var character in sub_str1)
                        {
                            int index = (int)char.ToUpper(character) - 64;
                            if (index < 10)
                            {
                                num += "0" + index.ToString();
                            }
                            else
                            {
                                num += index.ToString();
                            }

                            i++;
                        }
                        encoded.Add(num);
                    }
                    else
                    {
                        string sub_str2 = value.Substring(i, 3);
                        string num = "";
                        foreach (var character in sub_str2)
                        {
                            int index = (int)char.ToUpper(character) - 64;
                            if (index < 10)
                            {
                                num += "0" + index.ToString();
                            }
                            else
                            {
                                num += index.ToString();
                            }
                        }
                        i = i + 2;
                        encoded.Add(num);
                    }

                }
                return encoded;
            }
            public static void run_LSTM_LC1()
            {
                /*
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
                var type = CNTK.DataType.Float;
                var features = CNTKUtil.NetUtil.Var(new int[] { 2 }, DataType.Float);
                var labels = CNTKUtil.NetUtil.Var(new int[] { 1 }, DataType.Float);

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
                var numInputs = 45;
                TemporalMemory tm1 = new TemporalMemory();
                HomeostaticPlasticityController hpa = new HomeostaticPlasticityController(mem, 45, (isStable, numPatterns, actColAvg, seenInputs) =>
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
                            */
                /*
                 * Testing Model
                 */
                /*
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
                */
                /*                    cls.TraceState($"cellState_MinPctOverlDuty-{p[KEY.MIN_PCT_OVERLAP_DUTY_CYCLES]}_MaxBoost-{p[KEY.MAX_BOOST]}.csv");
                */
                /*
                Debug.WriteLine("---- Spatial Pooler column state  ----");
                Debug.WriteLine("------------ END ------------");
            */
            }
            public static List<Object> fetchEncoders(string[] classes)
            {
                Dictionary<string, object> settingsScalarEncoder_PassCount = new Dictionary<string, object>()
                {
                    { "W", 9},
                    { "N", 15},
                    { "Radius", -1.0},
                    { "MinVal", (double)1},
                    { "Periodic", true},
                    { "Name", "scalar"},
                    { "ClipInput", false},
                    { "MaxVal", (double)26}
                };
                ScalarEncoder encoder_PassCnt = new ScalarEncoder(settingsScalarEncoder_PassCount);



                Dictionary<String, Object> encoderSettings = new Dictionary<string, object>();
                encoderSettings.Add("W", 3);
                encoderSettings.Add("Radius", (double)1);
                CategoryEncoder encoder_Category = new CategoryEncoder(classes, encoderSettings);
                List<Object> result = new List<object>();
                result.Add(encoder_PassCnt);
                result.Add(encoder_Category);
                return result;

            }
            public static void RunExperiment_LC1(int inputBits, Parameters p, EncoderBase encoder, Dictionary<string, List<int[]>> inputValues, Dictionary<string, List<string>> fetched_Data)
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
                var numInputs = 20;
                TemporalMemory tm1 = new TemporalMemory();
                HomeostaticPlasticityController hpa = new HomeostaticPlasticityController(mem, numInputs, (isStable, numPatterns, actColAvg, seenInputs) =>
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
                }, numOfCyclesToWaitOnChange: 25); // Configuration -1
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
                for (int i = 0; i < 4; i++)
                {
                    if (activeColumnsLst.ContainsKey(i) == false)
                        activeColumnsLst.Add(i, new List<List<int>>());
                }
                int maxCycles = 500;
                int maxPrevInputs = 19;
                //int maxPrevInputs = 20;
                List<string> previousInputs = new List<string>();
                previousInputs.Add("");

                Debug.WriteLine(p.ToString());

            // Now training with SP+TM. SP is pretrained on the given input pattern.



            }

        }
}


