using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.IO;
using System.Diagnostics;
using System.Text.RegularExpressions;


using NeoCortexApi;
using NeoCortexApi.Encoders;
using NeoCortexApi.Utility;
using NeoCortexApi.Entities;
using NeoCortexApi.Classifiers;
using NeoCortexApi.Network;

using HtmImageEncoder;

using Daenet.ImageBinarizerLib.Entities;
using Daenet.ImageBinarizerLib;

namespace SimpleMultiSequenceLearning
{
    /// <summary>
    /// Helper Method For MultiSequence Learning
    /// </summary>
    public class MyHelperMethod
    {
        static readonly string[] SequenceClasses = new string[] { "inactive - exp", "mod. active", "very active", "inactive - virtual" };
        static readonly float[][] SequenceClassesOneHotEncoding = new float[][] { new float[] { 0, 0, 0, 1 }, new float[] { 0, 0, 1, 0 }, new float[] { 0, 1, 0, 0 }, new float[] { 1, 0, 0, 0 } };

        /// <summary>
        ///     Fetch Data Sequence from the File 
        /// </summary>
        /// <param name="dataFilePath"></param>
        /// <returns></returns>
        public static List<Dictionary<string, string>> ReadSequencesDataFromCSV(string dataFilePath)
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

        /// <summary>
        ///     Encoding Alphabetic Sequences
        /// </summary>
        /// <param name="trainingData"></param>
        /// <returns></returns>
        /// 
        public static List<Dictionary<string, int[]>> TrainEncodeSequencesFromCSV(List<Dictionary<string, string>> trainingData)
        {
            List<Dictionary<string, int[]>> ListOfEncodedTrainingSDR = new List<Dictionary<string, int[]>>();

            ScalarEncoder encoder_Alphabets = FetchAlphabetEncoder();

            foreach (var sequence in trainingData)
            {
                int keyForUniqueIndex = 0;
                var tempDictionary = new Dictionary<string, int[]>();

                foreach (var element in sequence)
                {
                    keyForUniqueIndex++;
                    var elementLabel = element.Key + "," + element.Value;
                    var elementKey = element.Key;
                    int[] sdr = new int[0];
                    sdr = sdr.Concat(encoder_Alphabets.Encode(char.ToUpper(element.Key.ElementAt(0)) - 64)).ToArray();

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

        /// <summary>
        /// After Alpha Sequence is Learnt, PredictInputSequence will carry out prediction of the Alphabets from the
        /// Sequence which is read from the sequence (CSV Folder) 
        /// </summary>
        /// <param name="list"></param>
        public static List<int[]> PredictInputSequence(string userInput, Boolean EncodeSingleAlphabet)
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


        /// <summary>
        ///         FetchAlphabetEncoder 
        /// </summary>
        /// <returns> SCALAR ENCODERS</returns>
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

        public void BinarizeImageTraining(string InputPath, string OutputPath, int height, int width)
        {
            if (Directory.Exists(InputPath))
            {
                // Initialize HTMModules 
                int inputBits = height * width;
                int numColumns = 1024;
                HtmConfig cfg = new HtmConfig(new int[] { inputBits }, new int[] { numColumns });
                var mem = new Connections(cfg);

                SpatialPoolerMT sp = new SpatialPoolerMT();
                sp.Init(mem);

                // For Apple
                if (Directory.Exists(Path.Join(InputPath, "Apple")))
                {
                    string[] directoryEntries = System.IO.Directory.GetFileSystemEntries(Path.Join(InputPath, "Apple"));

                    foreach (string directoryEntry in directoryEntries)
                    {
                        string filename = Path.GetFileName(directoryEntry);

                        string Outputfilename = Path.GetFileName(Path.Join(OutputPath, "Apple", $"Binarized_{Path.GetFileName(filename)}"));

                        ImageEncoder imageEncoder = new ImageEncoder(new BinarizerParams { InputImagePath = directoryEntry, OutputImagePath = Path.Join(OutputPath, "Apple"), ImageWidth = height, ImageHeight = width });

                        imageEncoder.EncodeAndSaveAsImage(directoryEntry, Outputfilename, "Png");

                        CortexLayer<object, object> layer1 = new CortexLayer<object, object>("L1");
                        layer1.HtmModules.Add("encoder", imageEncoder);
                        layer1.HtmModules.Add("sp", sp);

                        //Test Compute method
                        var computeResult = layer1.Compute(directoryEntry, true) as int[];
                        var activeCellList = GetActiveCells(computeResult);
                        Debug.WriteLine($"Active Cells computed from Image - Apple {filename}: {activeCellList}");
                    }
                    Console.WriteLine("Apple Training Finish");
                }
                else
                {
                    Console.WriteLine("Apple Directory Not Found");
                }

                // For Avocado
                if (Directory.Exists(Path.Join(InputPath, "Avocado")))
                {
                    string[] directoryEntries = System.IO.Directory.GetFileSystemEntries(Path.Join(InputPath, "Avocado"));

                    foreach (string directoryEntry in directoryEntries)
                    {
                        string filename = Path.GetFileName(directoryEntry);

                        string Outputfilename = Path.GetFileName(Path.Join(OutputPath, "Avocado", $"Binarized_{Path.GetFileName(filename)}"));

                        ImageEncoder imageEncoder = new ImageEncoder(new BinarizerParams { InputImagePath = directoryEntry, OutputImagePath = Path.Join(OutputPath, "Avocado"), ImageWidth = height, ImageHeight = width });

                        imageEncoder.EncodeAndSaveAsImage(directoryEntry, Outputfilename, "Png");

                        CortexLayer<object, object> layer1 = new CortexLayer<object, object>("L1");
                        layer1.HtmModules.Add("encoder", imageEncoder);
                        layer1.HtmModules.Add("sp", sp);

                        //Test Compute method
                        var computeResult = layer1.Compute(directoryEntry, true) as int[];
                        var activeCellList = GetActiveCells(computeResult);
                        Debug.WriteLine($"Active Cells computed from Image - Avocado {filename}: {activeCellList}");
                    }
                    Console.WriteLine("Avocado Training Finish");
                }
                else
                {
                    Console.WriteLine("Avocado Directory Not Found");
                }

                // For Banana
                if (Directory.Exists(Path.Join(InputPath, "Banana")))
                {
                    string[] directoryEntries = System.IO.Directory.GetFileSystemEntries(Path.Join(InputPath, "Banana"));

                    foreach (string directoryEntry in directoryEntries)
                    {
                        string filename = Path.GetFileName(directoryEntry);

                        string Outputfilename = Path.GetFileName(Path.Join(OutputPath, "Banana", $"Binarized_{Path.GetFileName(filename)}"));

                        ImageEncoder imageEncoder = new ImageEncoder(new BinarizerParams { InputImagePath = directoryEntry, OutputImagePath = Path.Join(OutputPath, "Banana"), ImageWidth = height, ImageHeight = width });

                        imageEncoder.EncodeAndSaveAsImage(directoryEntry, Outputfilename, "Png");

                        CortexLayer<object, object> layer1 = new CortexLayer<object, object>("L1");
                        layer1.HtmModules.Add("encoder", imageEncoder);
                        layer1.HtmModules.Add("sp", sp);

                        //Test Compute method
                        var computeResult = layer1.Compute(directoryEntry, true) as int[];
                        var activeCellList = GetActiveCells(computeResult);
                        Debug.WriteLine($"Active Cells computed from Image - Banana {filename}: {activeCellList}");
                    }
                    Console.WriteLine("Banana Training Finish");
                }
                else
                {
                    Console.WriteLine("Banana Directory Not Found");
                }
            }
            else
            {
                Console.WriteLine("Please check the Directory Path");
            }
        }


        /// <summary>
        ///     Fetch Data Sequence from the File  IN PROGRESS.......!
        /// </summary>
        /// <param name="dataFilePath"></param>
        /// <returns></returns>
        public static List<Dictionary<string, string>> ReadImageDataSetsFromFolder(string dataFilePath)
        {
            //List<Dictionary<string, string>> SequencesCollectedData = new List<Dictionary<string, string>>();

            //List<Dictionary<string, string>> SequencesCollection = new List<Dictionary<string, string>>();

            Dictionary<string, List<string>> SequencesCollection = new Dictionary<string, List<string>>();

            int keyForUniqueIndexes = 0;


            if (Directory.Exists(dataFilePath))
            {
                if (Directory.Exists(Path.Join(dataFilePath, "Apple")))
                {

                    String directoryEntries = Path.Join(dataFilePath, "Apple");

                    List<string> Apples = Directory.EnumerateFiles(directoryEntries).Select(d => new DirectoryInfo(d).Name).ToList();

                    SequencesCollection.Add("Apples", Apples);
                }
                if (Directory.Exists(Path.Join(dataFilePath, "Avocado")))
                {
                    String directoryEntries = Path.Join(dataFilePath, "Avocado");

                    List<string> Avocados = Directory.EnumerateFiles(directoryEntries).Select(d => new DirectoryInfo(d).Name).ToList();

                    SequencesCollection.Add("Avocado", Avocados);
                }
                if (Directory.Exists(Path.Join(dataFilePath, "Banana")))
                {
                    String directoryEntries = Path.Join(dataFilePath, "Banana");

                    List<string> Bananas = Directory.EnumerateFiles(directoryEntries).Select(d => new DirectoryInfo(d).Name).ToList();

                    SequencesCollection.Add("Banana", Bananas);
                }
            }
            return null;


        }


        /// <summary>
        /// Convert int array to string for better representation
        /// </summary>
        /// <param name="computeResult"></param>
        /// <returns></returns>
        /// <exception cref="System.NotImplementedException"></exception>
        private string GetActiveCells(int[] computeResult)
        {
            string result = String.Join(",", computeResult);
            return result;
        }
    }
}