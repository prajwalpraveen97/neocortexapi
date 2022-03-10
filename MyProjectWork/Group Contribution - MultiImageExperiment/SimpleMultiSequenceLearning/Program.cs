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
        /// TRAINING FILE PATH
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
            Console.WriteLine("######## ########    ###    ##     ##    ##    ##  #######   #######  ########  #### ########  ######  ");
            Console.WriteLine("   ##    ##         ## ##   ###   ###    ###   ## ##     ## ##     ## ##     ##  ##  ##       ##    ## ");
            Console.WriteLine("   ##    ##        ##   ##  #### ####    ####  ## ##     ## ##     ## ##     ##  ##  ##       ##       ");
            Console.WriteLine("   ##    ######   ##     ## ## ### ##    ## ## ## ##     ## ##     ## ########   ##  ######    ######  ");
            Console.WriteLine("   ##    ##       ######### ##     ##    ##  #### ##     ## ##     ## ##     ##  ##  ##             ## ");
            Console.WriteLine("   ##    ##       ##     ## ##     ##    ##   ### ##     ## ##     ## ##     ##  ##  ##       ##    ## ");
            Console.WriteLine("   ##    ######## ##     ## ##     ##    ##    ##  #######   #######  ########  #### ########  ######  ");

            Console.WriteLine("\n");
            Console.WriteLine("\n");
            Console.WriteLine("\n");

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


            Console.WriteLine("\n\n");
            Console.WriteLine("\n\n");

            Console.WriteLine("*********************************************************************************************");
            Console.WriteLine("***********************************   MACHINE LEARNING     **********************************");
            Console.WriteLine("***********************************   NEO - CORTEX API     **********************************");
            Console.WriteLine("***********************************   MULTI - SEQUENCE     **********************************");
            Console.WriteLine("***********************************       LEARNING         **********************************");
            Console.WriteLine("*********************************************************************************************");
            Console.WriteLine("*********************************************************************************************");

            Console.WriteLine("\n\n");
            Console.WriteLine("\n\n");
            Console.WriteLine("Variables are being trained Please Wait....");

            Console.WriteLine("Training Model In Progress.....");
            // RunMultiSimpleSequenceLearningExperiment();

            // RunMultiSequenceLearningExperiment(SequenceDataFile);

            MyHelperMethod MultiSequenceForImage = new MyHelperMethod();
            MultiSequenceForImage.BinarizeImage(InputPicPath, OutputPicPath);

            int width = 30;
            int height = 30;
            //MultiSequenceForImage.LearningInLayer(width, height, OutputPicPath);



            Console.WriteLine("EncodeAndSaveAsImage.....");
        }

        private static void RunMultiSequenceLearningExperiment(string datafilepath)
        {
            Dictionary<string, List<double>> sequences = new Dictionary<string, List<double>>();

            //sequences.Add("S1", new List<double>(new double[] { 0.0, 1.0, 0.0, 2.0, 3.0, 4.0, 5.0, 6.0, 5.0, 4.0, 3.0, 7.0, 1.0, 9.0, 12.0, 11.0, 12.0, 13.0, 14.0, 11.0, 12.0, 14.0, 5.0, 7.0, 6.0, 9.0, 3.0, 4.0, 3.0, 4.0, 3.0, 4.0 }));
            //sequences.Add("S2", new List<double>(new double[] { 0.8, 2.0, 0.0, 3.0, 3.0, 4.0, 5.0, 6.0, 5.0, 7.0, 2.0, 7.0, 1.0, 9.0, 11.0, 11.0, 10.0, 13.0, 14.0, 11.0, 7.0, 6.0, 5.0, 7.0, 6.0, 5.0, 3.0, 2.0, 3.0, 4.0, 3.0, 4.0 }));

            //sequences.Add("S1", new List<double>(new double[] { 2.0, 4.0, 6.0, 8.0,  10.0, 12.0}));
            //sequences.Add("S2", new List<double>(new double[] { 3.0, 6.0, 9.0, 12.0, 15.0, 18.0}));

            //sequences.Add("S1", new List<double>(new double[] { 0.0, 1.0, 3.0, 5.0, 7.0 }));
            //sequences.Add("S2", new List<double>(new double[] { 2.0, 4.0, 6.0, 8.0, 10.00 }));

            // List of Prime Numbers from 0 to 100
            //sequences.Add("S1", new List<double>(new double[] { 2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37}));

            //sequences.Add("S3", new List<double>(new double[] { 4, 6, 8, 9, 10, 12, 14, 15, 16 }));
            //sequences.Add("S2", new List<double>(new double[] { 41, 43, 47,  53, 59, 61, 67, 71,73, 79, 83, 89, 97 }));


            // List of Composite Numbers from 1 to 100          { F, A, K, L , M, 12, 14, 15,16 }));
            //sequences.Add("S3", new List<double>(new double[] { 4, 6, 8, 9, 10, 12, 14, 15,16 }));
            //sequences.Add("S4", new List<double>(new double[] {18, 20, 21, 22, 24, 25,26, 27, 28 }));
            //sequences.Add("S5", new List<double>(new double[] {30, 32, 33, 34, 35, 36, 38, 39, 40 }));
            //sequences.Add("S6", new List<double>(new double[] {42, 44, 45, 46, 48, 49, 50, 51, 52 }));
            // sequences.Add("S7", new List<double>(new double[] {54, 55, 56, 57, 58, 60, 62, 63, 64 }));
            // sequences.Add("S8", new List<double>(new double[] {65, 66, 68, 69, 70, 72, 74, 75, 76 }));
            // sequences.Add("S9", new List<double>(new double[] {77, 78, 80, 81, 82, 84, 85, 86, 87 }));
            // sequences.Add("S10", new List<double>(new double[] {88, 90, 91, 92, 93, 94, 95, 96, 98, 99, 100 }));

            sequences.Add("TwoMultiple", new List<double>(new double[] { 2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0 }));

            sequences.Add("ThreeMultiple", new List<double>(new double[] { 3.0, 6.0, 9.0, 12.0, 15.0, 18.0, 21.0, 24.0 }));

            sequences.Add("FiveMultiple", new List<double>(new double[] { 5.0, 10.0, 15.0, 20.0, 25.0, 30.0, 35.0, 40.0 }));

            sequences.Add("SevenMultiple", new List<double>(new double[] { 7.0, 14.0, 21.0, 28.0, 35.0, 42.0, 49.0 }));

            sequences.Add("ElevenMultiple", new List<double>(new double[] { 11.0, 22.0, 33.0, 44.0 }));


            //Dictionary<string, List<string>> GauravSequence = new Dictionary<string, List<string>>();

            //GauravSequence.Add("Sequence1", new List<string>(new string[] { "123456789"}));

            var trainingData = MyHelperMethod.ReadSequencesDataFromCSV(datafilepath);
            var trainingDataProcessed = MyHelperMethod.TrainEncodeSequencesFromCSV(trainingData);


            //
            // Prototype for building the prediction engine.
            MultiSequenceLearning experiment = new MultiSequenceLearning();


            var trained_HTM_model = experiment.RunAlphabetsLearning(trainingDataProcessed, true);


            var trained_CortexLayer = trained_HTM_model.Keys.ElementAt(0);
            var trained_Classifier = trained_HTM_model.Values.ElementAt(0);


            //var predictor = experiment.Run(sequences);

            Console.WriteLine("Ready to Predict.....");

            /*var list1 = new double[] { 1.0, 2.0, 3.0 };
            var list2 = new double[] { 2.0, 3.0, 4.0 };
            var list3 = new double[] { 8.0, 1.0, 2.0 };
            var list4 = new double[] { 5.0, 1.0, 7.0 };
*/
            // replace with alphabets 
            // own encoder for letter
            // change letter to scalar encoder
            // note : can use ascii value

            // var list2 = new double[] { 2.0, 3.0, 5.0, 11.0 };
            //var list2 = new double[] { 4.0, 6.0, 9.0, 15.0, 21.0, 7.0 , 30.0};

            int BufferSize;

/*            Console.WriteLine("Enter Total number of sequences you want to give..");

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
            }*/

            Console.WriteLine("\n");
            /*predictor.Reset();
            PredictNextElement(predictor, buffer);*/

            /* PredictNextElement(predictor, list4);

               predictor.Reset();
               PredictNextElement(predictor, list1);

               predictor.Reset();
               PredictNextElement(predictor, list2);

               predictor.Reset();
               PredictNextElement(predictor, list3);
               */


            Console.WriteLine("ENTER CANCER SEQUENCE:            *note format->AAAAVVV {AlphabeticSequence}");
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
                var possibleClass = "";
                foreach (var class_ in Classcounts)
                {
                    Console.WriteLine($"Predicted Class : {class_.possibleClass.Split("_")[0]} \t votes: {class_.Count}");
                }


                Console.WriteLine("ENTER NEXT SEQUENCE :");
                userInput = Console.ReadLine();

            }
        }

        private static void PredictNextElement(HtmPredictionEngine predictor, double[] list)
        {
            Console.WriteLine("-------------------Start of PredictNextElement Function------------------------");

            foreach (var item in list)
            {
                // {1.0,2.0,3.0}
                var res = predictor.Predict(item);
                if (res.Count > 0)
                {
                    foreach (var pred in res)
                    {
                        //Console.WriteLine($"PredictedInput = {pred.PredictedInput}");
                        //Console.WriteLine($"Similarity     = {pred.Similarity}");

                        Debug.WriteLine($"PredictedInput = {pred.PredictedInput} <---> Similarity = {pred.Similarity}\n");
                    }
                    //Token   -->  tokens[0] S1
                    //             tokens[1] 3-4-2-5-0-1-2

                    // Tokens2 --> tokens2[0] = S1_3
                    //             tokens2[1] = 4    <-- Everytime this array element is checked
                    //             tokens2[2] = 2
                    //             tokens2[3] = 5
                    //             tokens2[4] = 0
                    //             tokens2[5] = 1
                    //             tokens2[6] = 2
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
