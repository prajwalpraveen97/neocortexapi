using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace SimpleMultiSequenceLearning
{
    public class StartupCode
    {
        /// <summary>
        /// Training File Paths For Images and Sequences
        /// </summary>
        /// 
        static readonly string SequenceDataFile = Path.GetFullPath(Directory.GetParent(Environment.CurrentDirectory).Parent.Parent.FullName + @"\TrainingFiles\TrainingFile.csv");

        static readonly string InputPicPath = Path.GetFullPath(Directory.GetParent(Environment.CurrentDirectory).Parent.Parent.FullName + @"\InputFolder\");

        static readonly string OutputPicPath = Path.GetFullPath(Directory.GetParent(Environment.CurrentDirectory).Parent.Parent.FullName + @"\BinarizedImage\");

        /// <summary>
        /// Print Message During Startup of Program
        /// </summary>
        /// <param name="None"></param>
        public void PrintDebugMessageOnStartUp()
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
        }

        /// <summary>
        /// Start MultiSequence Learning Program
        /// Option 1 for Numbers
        /// Option 2 for Alphabet
        /// Option 3 for Images
        /// </summary>
        /// <param name="Userinput"></param>
        public void MultiSequenceLearning(int Userinput)
        {

            switch (Userinput)
            {
                case 1:
                    {
                        Console.WriteLine("User Selected MultiSequence Experiment - Numbers\n");
                        HelperMethod_Numbers multiSeqLearn_Numbers = new HelperMethod_Numbers();
                        multiSeqLearn_Numbers.MultiSequenceLearning_Numbers();
                    }
                    break;

                case 2:
                    {
                        Console.WriteLine("User Selected MultiSequence Experiment - Alphabets\n");
                        HelperMethod_Alphabets multiSeqLearn_Alphabets = new HelperMethod_Alphabets();

                        multiSeqLearn_Alphabets.MultiSequenceLearning_Alphabets(SequenceDataFile);
                    }
                    break;

                case 3:
                    {
                        Console.WriteLine("User Selected MultiSequence Experiment - Image");

                        HelperMethod_Images MultiSequenceForImage = new HelperMethod_Images();

                        int height = 40;
                        int width = 40;

                        MultiSequenceForImage.MultiSequenceLearning_Images(InputPicPath, OutputPicPath, height, width);
                    }
                    break;

                default:
                    {
                        Console.WriteLine("User Entered Invalid Option");
                    }
                    break;

            }
        }
    }
}
