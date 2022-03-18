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
    public class MultiSeqLearn_Numbers
    {
        /// <summary>
        /// Runs MultiSequence Learning Experiment - To Carry out Sequence Learning for Numbers.
        /// </summary>
        public void MultiSequenceLearning_Numbers()
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
        /// After Number Sequence is Learnt, PredictNextElement will carry out prediction of the elements from the
        /// Sequence which is input from the user 
        /// </summary>
        /// <param name="list"></param>
        public static void PredictNextElement(HtmPredictionEngine predictor, double[] list)
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
