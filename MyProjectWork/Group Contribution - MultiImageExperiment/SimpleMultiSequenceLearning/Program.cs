using NeoCortexApi;
using NeoCortexApi.Classifiers;
using NeoCortexApi.Entities;
using NeoCortexApi.Network;
using NeoCortexApi.Encoders;
using NeoCortexApi.Utility;

using HtmImageEncoder;

using Daenet.ImageBinarizerLib.Entities;
using Daenet.ImageBinarizerLib;

using Newtonsoft.Json;

using SkiaSharp;

using System;
using System.IO;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Globalization;
using System.Drawing;

using static SimpleMultiSequenceLearning.MultiSequenceLearning;


namespace SimpleMultiSequenceLearning
{
    class Program
    {
        /// <summary>
        /// Training File Paths For Images and Sequences
        /// </summary>
        /// 
        static readonly string SequenceDataFile = Path.GetFullPath(Directory.GetParent(Environment.CurrentDirectory).Parent.Parent.FullName + @"\TrainingFiles\TrainingFile.csv");

        static readonly string InputPicPath = Path.GetFullPath(Directory.GetParent(Environment.CurrentDirectory).Parent.Parent.FullName + @"\InputFolder\");

        static readonly string OutputPicPath = Path.GetFullPath(Directory.GetParent(Environment.CurrentDirectory).Parent.Parent.FullName + @"\BinarizedImage\");


        /// <summary>
        /// This sample shows a typical experiment code for SP and TM.
        /// You must start this code in debugger to follow the trace.
        /// and TM.
        /// </summary>
        /// <param name="args"></param>
        /// 
        static void Main(string[] args)
        {
            // Set the Foreground color to blue
            Console.ForegroundColor = ConsoleColor.DarkYellow;
            Console.WriteLine("######## ########    ###    ##     ##    ##    ##  #######   #######  ########  #### ########  ######  ");
            Console.WriteLine("   ##    ##         ## ##   ###   ###    ###   ## ##     ## ##     ## ##     ##  ##  ##       ##    ## ");
            Console.WriteLine("   ##    ##        ##   ##  #### ####    ####  ## ##     ## ##     ## ##     ##  ##  ##       ##       ");
            Console.WriteLine("   ##    ######   ##     ## ## ### ##    ## ## ## ##     ## ##     ## ########   ##  ######    ######  ");
            Console.WriteLine("   ##    ##       ######### ##     ##    ##  #### ##     ## ##     ## ##     ##  ##  ##             ## ");
            Console.WriteLine("   ##    ##       ##     ## ##     ##    ##   ### ##     ## ##     ## ##     ##  ##  ##       ##    ## ");
            Console.WriteLine("   ##    ######## ##     ## ##     ##    ##    ##  #######   #######  ########  #### ########  ######  ");

            Console.WriteLine("\n\n\n");

            // Set the Foreground color to blue
            Console.ForegroundColor = ConsoleColor.DarkRed;

            Console.WriteLine("             :::::::::::::::::::...:~!777!!~^:....:::::::::::::");
            Console.WriteLine("             :::::::::::::::::..^?G##########BPJ~:..:::::::::::");
            Console.WriteLine("             ::::::::::::::::.:J&&BGPPPPPPPPGGB#&BJ^.::::::::::");
            Console.WriteLine("             :::::::::::::::.~B@BPPPPPPPPPPPPPPPPB&#?..::::::::");
            Console.WriteLine("             ::::::::::::::.~#@GPPPPPPGBB##########@@5:..::::::");
            Console.WriteLine("             :::::::::::::.^#@BPPPPG#&#GP5YYJ?????JYG##P~.:::::");
            Console.WriteLine("             ::::::::......P@#PPPPP&@G5!~~~~:       .^?#@?.::::");
            Console.WriteLine("             ::::::.::~!7?Y@&GPPPPG@#55Y!~~~~~^:::.::^~!&&^.:::");
            Console.WriteLine("             :::::.?B##&##@@BGPPPPG@&555Y?!!!!!!!!!!!7?J#@7.:::");
            Console.WriteLine("             ::::.7@&GPPPP&@GGPPPPP#@BP55555YYYYYY55555P@&^.:::");
            Console.WriteLine("             :::.^#@GGGGGB@&GGPPPPPPB&&#BGGGPPPPPPPPPGB@#!.::::");
            Console.WriteLine("             :::.?@&GGGGGB@#GGGPPPPPPPGBB##&&&&&&&&&&#@@~.:::::");
            Console.WriteLine("             :::.5@BGGGGG#@#GGGPPPPPPPPPPPPPPPPPPPPPPP&&^.:::::");
            Console.WriteLine("             :::.P@BGGGGG#@#GGGGPPPPPPPPPPPPPPPPPPPPPG@#:.:::::");
            Console.WriteLine("             :::.P@BGGGGG#@#GGGGPPPPPPPPPPPPPPPPPPPPPG@G.::::::");
            Console.WriteLine("             :::.5@#GGGGG#@#GGGGGPPPPPPPPPPPPPPPPPPPPB@5.::::::");
            Console.WriteLine("             :::.?@#GGGGGB@#GGGGGGGGPPPPPPPPPPPPPPPPG#@?.::::::");
            Console.WriteLine("             :::.~@&GGGGGB@&GGGGGGGGGGGGGPPPPPPPGGGGG&@~.::::::");
            Console.WriteLine("             ::::.G@BGGGGG@&GGGGGGGGGGGGGGGGGGGGGGGGB@B:.::::::");
            Console.WriteLine("             ::::.~G&&###&@@BGGGGGGGGGB########BGGGG#@Y.:::::::");
            Console.WriteLine("             :::::.:~7???7G@BGGGGGGGGB@&YYP@&BBGGGGG&@!.:::::::");
            Console.WriteLine("             :::::::......7@&GGGGGGGGB@G..~@&GGGGGGB@#:.:::::::");
            Console.WriteLine("             ::::::::::::.^&@GGGGGGGGB@5..:#@BGGGGG#@Y.::::::::");
            Console.WriteLine("             :::::::::::::.Y@&&######&@J.:.5@&&&&&&&#~.::::::::");
            Console.WriteLine("             :::::::::::::::~!?JJYJJ?7~:::.^!!!!!!!~:.:::::::::");


            Console.WriteLine("\n\n\n\n");

            // Set the Foreground color to blue
            Console.ForegroundColor = ConsoleColor.DarkBlue;
            Console.WriteLine("*********************************************************************************************");
            Console.WriteLine("***********************************   MACHINE LEARNING     **********************************");
            Console.WriteLine("***********************************   NEO - CORTEX API     **********************************");
            Console.WriteLine("***********************************   MULTI - SEQUENCE     **********************************");
            Console.WriteLine("***********************************       LEARNING         **********************************");
            Console.WriteLine("*********************************************************************************************");
            Console.WriteLine("*********************************************************************************************");

            Console.WriteLine("\n\n\n\n");

            Console.WriteLine("**************             Multi Sequence Learning               ************** ");
            Console.WriteLine("**************  Option - 1 Multi Sequence Learning - Numbers     ************** ");
            Console.WriteLine("**************  Option - 2 Multi Sequence Learning - Alphabets   ************** ");
            Console.WriteLine("**************  Option - 3 Multi Sequence Learning - Image       ************** ");

            Console.WriteLine("\n");
            Console.WriteLine("Please Enter An Option to Continue with MultiSequence Experiment");

            // Set the Foreground color to blue
            Console.ForegroundColor = ConsoleColor.White;

            int Option = Convert.ToInt16(Console.ReadLine());

            if (Option == 1)
            {
                Console.WriteLine("User Selected MultiSequence Experiment - Numbers\n");
                MultiSequenceLearning_Numbers();
            }
            if (Option == 2)
            {
                Console.WriteLine("User Selected MultiSequence Experiment - Alphabets\n");
                MultiSequenceLearning_Alphabets(SequenceDataFile);
            }
            if (Option == 3)
            {
                Console.WriteLine("User Selected MultiSequence Experiment - Image");

                MyHelperMethod MultiSequenceForImage = new MyHelperMethod();
                int imageheight = 100;
                int imagewidth = 100;

                var trainingData2 = MyHelperMethod.ReadImageDataSetsFromFolder(InputPicPath);
                MultiSequenceForImage.BinarizeImageTraining(InputPicPath, OutputPicPath, imageheight, imagewidth);
            }
        }

        /// <summary>
        /// Runs MultiSequence Learning Experiment - To Carry out Sequence Learning for Numbers.
        /// </summary>
        private static void MultiSequenceLearning_Numbers()
        {
            Dictionary<string, List<double>> sequences = new Dictionary<string, List<double>>();

            sequences.Add("TwoMultiple", new List<double>(new double[] { 2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0 }));
            sequences.Add("ThreeMultiple", new List<double>(new double[] { 3.0, 6.0, 9.0, 12.0, 15.0, 18.0, 21.0, 24.0 }));
            sequences.Add("FiveMultiple", new List<double>(new double[] { 5.0, 10.0, 15.0, 20.0, 25.0, 30.0, 35.0, 40.0 }));
            sequences.Add("SevenMultiple", new List<double>(new double[] { 7.0, 14.0, 21.0, 28.0, 35.0, 42.0, 49.0 }));
            sequences.Add("ElevenMultiple", new List<double>(new double[] { 11.0, 22.0, 33.0, 44.0 }));

            MultiSequenceLearning experiment = new MultiSequenceLearning();

            Console.WriteLine("Variables are being trained Please Wait....");

            var predictor = experiment.Run(sequences);

            Console.WriteLine("Ready to Predict.....");

            int BufferSize;
            Console.WriteLine("Enter Total number of sequences you want to give..");
            BufferSize = Convert.ToInt32(Console.ReadLine());

            Console.WriteLine($"Array Size is : {BufferSize}");
            Console.WriteLine("Enter Sequence of Numbers to be Predicted....");

            double[] buffer = new double[BufferSize];

            for (int i = 0; i < BufferSize; i++)
            {
                buffer[i] = Convert.ToDouble(Console.ReadLine());
            }

            Console.WriteLine($"Entered Number are : ");
            for (int i = 0; i < BufferSize; i++)
            {
                Console.Write("{0}", buffer[i]);
                Console.Write("\t");
            }
            Console.WriteLine("\n");

            predictor.Reset();
            PredictNextElement(predictor, buffer);
        }


        /// <summary>
        /// Runs MultiSequence Learning Experiment - To Carry out Sequence Learning for Alphabets.
        /// </summary>
        /// <param name="datafilepath"></param>
        private static void MultiSequenceLearning_Alphabets(string datafilepath)
        {
            var trainingData = MyHelperMethod.ReadSequencesDataFromCSV(datafilepath);
            var trainingDataProcessed = MyHelperMethod.TrainEncodeSequencesFromCSV(trainingData);

            /// <summary>
            /// Prototype for building the prediction engine.
            ///  </summary>
            MultiSequenceLearning experiment = new MultiSequenceLearning();

            Console.WriteLine("Variables are being trained Please Wait....");

            var trained_HTM_model = experiment.RunAlphabetsLearning(trainingDataProcessed, true);
            var trained_CortexLayer = trained_HTM_model.Keys.ElementAt(0);
            var trained_Classifier = trained_HTM_model.Values.ElementAt(0);

            Console.WriteLine("Ready to Predict.....");

            Console.WriteLine("Enter Cancer Sequence:   *note format->AAAAVVV {AlphabeticSequence}");
            var userInput = Console.ReadLine();
            while (!userInput.Equals("q") && userInput != "Q")
            {
                var ElementSDRs = MyHelperMethod.PredictInputSequence(userInput, false);
                List<string> possibleClasses = new List<string>();

                for (int i = 0; i < userInput.Length; i++)
                {

                    var element = userInput.ElementAt(i);
                    var elementSDR = MyHelperMethod.PredictInputSequence(element.ToString(), true);

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
                                if (prediction.PredictedInput.Split(",")[0] == nextElementString)
                                {
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
                foreach (var class_ in Classcounts)
                {
                    Console.WriteLine($"Predicted Class : {class_.possibleClass.Split("_")[0]} \t votes: {class_.Count}");
                }
                Console.WriteLine("Enter Next Sequence :");
                userInput = Console.ReadLine();
            }
        }

        /// <summary>
        /// After Number Sequence is Learnt, PredictNextElement will carry out prediction of the elements from the
        /// Sequence which is input from the user 
        /// </summary>
        /// <param name="list"></param>
        private static void PredictNextElement(HtmPredictionEngine predictor, double[] list)
        {
            Console.WriteLine("-------------------Start of PredictNextElement Function------------------------");

            foreach (var item in list)
            {
                var res = predictor.Predict(item);
                if (res.Count > 0)
                {
                    foreach (var pred in res)
                    {
                        Debug.WriteLine($"PredictedInput = {pred.PredictedInput} <---> Similarity = {pred.Similarity}\n");
                    }
                    var tokens = res.First().PredictedInput.Split('_');
                    var tokens2 = res.First().PredictedInput.Split('-');
                    Console.WriteLine($"Predicted Sequence: {tokens[0]}, predicted next element {tokens2[tokens.Length - 1]}\n");
                }
                else
                    Console.WriteLine("Invalid Match..... \n");
            }
            Console.WriteLine("------------------------End of PredictNextElement ------------------------");
        }
    }
}