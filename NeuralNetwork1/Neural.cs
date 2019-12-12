using System;
using System.Collections.Generic;
using System.Linq;
using System.Diagnostics;
using System.Text;
using System.Threading.Tasks;
using System.Drawing;
using System.Windows.Forms;
using System.Collections;

namespace NeuralNetwork1
{
    /// <summary>
    /// Класс для хранения образа – входной массив сигналов на сенсорах, выходные сигналы сети, и прочее
    /// </summary>
    public class Sample
    {
        /// <summary>
        /// Входной вектор
        /// </summary>
        public double[] input = null;

        /// <summary>
        /// Выходной вектор, задаётся извне как результат распознавания
        /// </summary>
        public double[] output = null;

        /// <summary>
        /// Вектор ошибки, вычисляется по какой-нибудь хитрой формуле
        /// </summary>
        public double[] error = null;

        /// <summary>
        /// Действительный класс образа. Указывается учителем
        /// </summary>
        public FigureType actualClass;

        /// <summary>
        /// Распознанный класс - определяется после обработки
        /// </summary>
        public FigureType recognizedClass;

        /// <summary>
        /// Конструктор образа - на основе входных данных для сенсоров, при этом можно указать класс образа, или не указывать
        /// </summary>
        /// <param name="inputValues"></param>
        /// <param name="sampleClass"></param>
        public Sample(double[] inputValues, int classesCount, FigureType sampleClass = FigureType.Undef)
        {
            //  Клонируем массивчик
            input = (double[])inputValues.Clone();
            output = new double[classesCount];
            if (sampleClass != FigureType.Undef) output[(int)sampleClass] = 1;


            recognizedClass = FigureType.Undef;
            actualClass = sampleClass;
        }

        /// <summary>
        /// Обработка реакции сети на данный образ на основе вектора выходов сети
        /// </summary>
        public void processOutput()
        {
            if (error == null)
                error = new double[output.Length];

            //  Нам так-то выход не нужен, нужна ошибка и определённый класс
            recognizedClass = 0;
            for (int i = 0; i < output.Length; ++i)
            {
                error[i] = ((i == (int)actualClass ? 1 : 0) - output[i]);
                if (output[i] > output[(int)recognizedClass]) recognizedClass = (FigureType)i;
            }
        }

        /// <summary>
        /// Вычисленная суммарная квадратичная ошибка сети. Предполагается, что целевые выходы - 1 для верного, и 0 для остальных
        /// </summary>
        /// <returns></returns>
        public double EstimatedError()
        {
            double Result = 0;
            for (int i = 0; i < output.Length; ++i)
                Result += Math.Pow(error[i], 2);
            return Result;
        }

        /// <summary>
        /// Добавляет к аргументу ошибку, соответствующую данному образу (не квадратичную!!!)
        /// </summary>
        /// <param name="errorVector"></param>
        /// <returns></returns>
        public void updateErrorVector(double[] errorVector)
        {
            for (int i = 0; i < errorVector.Length; ++i)
                errorVector[i] += error[i];
        }

        /// <summary>
        /// Представление в виде строки
        /// </summary>
        /// <returns></returns>
        public override string ToString()
        {
            string result = "Sample decoding : " + actualClass.ToString() + "(" + ((int)actualClass).ToString() + "); " + Environment.NewLine + "Input : ";
            for (int i = 0; i < input.Length; ++i) result += input[i].ToString() + "; ";
            result += Environment.NewLine + "Output : ";
            if (output == null) result += "null;";
            else
                for (int i = 0; i < output.Length; ++i) result += output[i].ToString() + "; ";
            result += Environment.NewLine + "Error : ";

            if (error == null) result += "null;";
            else
                for (int i = 0; i < error.Length; ++i) result += error[i].ToString() + "; ";
            result += Environment.NewLine + "Recognized : " + recognizedClass.ToString() + "(" + ((int)recognizedClass).ToString() + "); " + Environment.NewLine;


            return result;
        }

        /// <summary>
        /// Правильно ли распознан образ
        /// </summary>
        /// <returns></returns>
        public bool Correct() { return actualClass == recognizedClass; }
    }

    /// <summary>
    /// Выборка образов. Могут быть как классифицированные (обучающая, тестовая выборки), так и не классифицированные (обработка)
    /// </summary>
    public class SamplesSet : IEnumerable
    {
        /// <summary>
        /// Накопленные обучающие образы
        /// </summary>
        public List<Sample> samples = new List<Sample>();

        /// <summary>
        /// Добавление образа к коллекции
        /// </summary>
        /// <param name="image"></param>
        public void AddSample(Sample image)
        {
            samples.Add(image);
        }
        public int Count { get { return samples.Count; } }

        public IEnumerator GetEnumerator()
        {
            return samples.GetEnumerator();
        }

        /// <summary>
        /// Реализация доступа по индексу
        /// </summary>
        /// <param name="i"></param>
        /// <returns></returns>
        public Sample this[int i]
        {
            get { return samples[i]; }
            set { samples[i] = value; }
        }

        public double ErrorsCount()
        {
            double correct = 0;
            double wrong = 0;
            foreach (var sample in samples)
                if (sample.Correct()) ++correct; else ++wrong;
            return correct / (correct + wrong);
        }
    }

    public class NeuralNetwork : BaseNetwork
    {
        /// Один нейрон
        private class Node
        {
            /// Входной взвешенный сигнал нейрона
            public double inputSignal = 0;
            /// Выходной сигнал нейрона
            public double outputSignal = 0;




            /// Количество узлов на предыдущем слое
            public int inputLayerSize = 0;
            /// Вектор входных весов нейрона
            public double[] weights = null;
            /// Вес на сигнале поляризации
            public double biasWeight = 0.01;
            /// Ссылка на предыдущий слой нейронов
            public Node[] inputLayer = null;
            /// Ошибка для данного нейрона
            public double error = 0;




            public static double biasSignal = -1.0;
            private static double initMinWeight = -1;
            private static double initMaxWeight = 1;

          
            /// Создаём один нейрон сети свектором входящих весов
            public Node(Node[] prevLayerNodes)
            {
                inputLayer = prevLayerNodes;  

                if (prevLayerNodes == null) return;

                inputLayerSize = prevLayerNodes.Length;
                weights = new double[inputLayerSize];

                for (int i = 0; i < weights.Length; ++i)
                    weights[i] = initMinWeight + (new Random()).NextDouble() * (initMaxWeight - initMinWeight);
            }


            public void Activate()
            {
                inputSignal = biasWeight * biasSignal;
                for (int i = 0; i < inputLayer.Length; ++i)
                    inputSignal += inputLayer[i].outputSignal * weights[i];
                outputSignal = ActivationFunction(inputSignal);
                inputSignal = 0;
            }

            public void backPrError(double ita)
            {
                error *= outputSignal * (1 - outputSignal);
                biasWeight += ita * error * biasSignal;
                for (int i = 0; i < inputLayerSize; i++)
                    inputLayer[i].error += error * weights[i];
                for (int i = 0; i < inputLayerSize; i++)
                    weights[i] += ita * error * inputLayer[i].outputSignal;
                error = 0;
            }

            public static double ActivationFunction(double inp)
            {
                return 1 / (1 + Math.Exp(-inp));
            }
        }





        public double LearningSpeed = 0.01;
        private Node[] Sensors;
        private Node[][] Layers;  
        private Node[] Outputs;

        public override void ReInit(int[] structure, double d = 0)
        {
            if (structure.Length < 2)
                throw new Exception("Ошибочная структура сети!");

            Layers = new Node[structure.Length][];

            Layers[0] = new Node[structure[0]];
            for (int neuron = 0; neuron < structure[0]; ++neuron)
                Layers[0][neuron] = new Node(null);
            Sensors = Layers[0];

            for (int layer = 1; layer < structure.Length; ++layer)
            {
                Layers[layer] = new Node[structure[layer]];   
                for (int neuron = 0; neuron < structure[layer]; ++neuron)
                    Layers[layer][neuron] = new Node(Layers[layer - 1]); 
            }
            Outputs = Layers[Layers.Length - 1];
        }

        public NeuralNetwork(int[] structure)
        {
            ReInit(structure);
        }



        // Прямой прогон сети
        private void Run(Sample image)
        {
   

            for (int i = 0; i < image.input.Length; i++)
                Sensors[i].outputSignal = image.input[i];

            for (int i = 1; i < Layers.Length; i++)
                for (int j = 0; j < Layers[i].Length; j++)
                    Layers[i][j].Activate();

            for (int i = 0; i < Layers[Layers.Length - 1].Length; i++)
                image.output[i] = Layers[Layers.Length - 1][i].outputSignal;

            image.processOutput();  
        }

        
        // Прогон ошибки
        private void BackProp(Sample image, double ita)
        {
            for (int i = 0; i < Layers[Layers.Length - 1].Length; i++)
                Layers[Layers.Length - 1][i].error = image.error[i];

            for (int i = Layers.Length - 1; i >= 0; --i)
                for (int j = 0; j < Layers[i].Length; ++j)
                    Layers[i][j].backPrError(ita);
        }

      
        // Распознавание образа
        public override FigureType Predict(Sample sample)
        {
            Run(sample);
            return sample.recognizedClass;
        }

        // Обучение одному заданному образу
        public override int Train(Sample sample, bool b =true)
        {
            int iters = 0;
            while (iters < 150)
            {
                Run(sample);
                if (sample.EstimatedError() < 0.2 && sample.Correct())
                {
                    Debug.WriteLine("Количество итераций = " + iters.ToString());
                    return iters;
                }

                ++iters;
                BackProp(sample, LearningSpeed);
            }
            if (iters == 150) Debug.WriteLine("Большое количество итераций!");
            return iters;
        }

        // Выходные значения
        public override double[] getOutput()
        {
            return Outputs.Select(n => n.outputSignal).ToArray();
        }

        
  
        public override double TrainOnDataSet(SamplesSet samplesSet, int epochs_count, double acceptable_erorr, bool b=true)
        {
            double guessLevel = 0;
            while (epochs_count > 0)
            {
                guessLevel = 0;
                for (int i = 0; i < samplesSet.samples.Count; ++i)
                    if (Train(samplesSet.samples.ElementAt(i)) == 0)
                        guessLevel += 1;
                guessLevel /= samplesSet.samples.Count;
                if (guessLevel > acceptable_erorr) return guessLevel;
                epochs_count--;
            };

            return guessLevel;
        }

        public override double TestOnDataSet(SamplesSet testSet)
        {
            if (testSet.Count == 0) return double.NaN;

            double guessLevel = 0;
            for (int i = 0; i < testSet.Count; ++i)
            {
                Sample s = testSet.samples.ElementAt(i);
                Predict(s);
                if (s.Correct()) guessLevel += 1;
            }
            return guessLevel / testSet.Count;
        }

        }
    
}
