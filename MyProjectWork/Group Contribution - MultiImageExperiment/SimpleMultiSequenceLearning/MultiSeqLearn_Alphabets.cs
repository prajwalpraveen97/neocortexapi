using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.IO;
using System.Diagnostics;
using System.Linq;
using System.Globalization;
using System.Drawing;


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

using static SimpleMultiSequenceLearning.MultiSequenceLearning;

namespace SimpleMultiSequenceLearning
{
    public class MultiSeqLearn_Alphabets
    {
        /// <summary>
        /// Runs MultiSequence Learning Experiment - To Carry out Sequence Learning for Alphabets.
        /// </summary>
        /// <param name="datafilepath"></param>
        public void MultiSequenceLearning_Alphabets(string datafilepath)
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


    }
}
