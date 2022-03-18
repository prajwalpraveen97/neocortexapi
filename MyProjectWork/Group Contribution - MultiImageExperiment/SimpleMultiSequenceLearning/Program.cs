using System;
using System.IO;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;

namespace SimpleMultiSequenceLearning
{
    class Program
    {
        /// <summary>
        /// Main Program Start
        /// </summary>
        /// 
        static void Main(string[] args)
        {
            StartupCode startup = new StartupCode();
            startup.PrintDebugMessageOnStartUp();

            int Option = Convert.ToInt16(Console.ReadLine());

            startup.MultiSequenceLearning(Option);
        }
    }
}