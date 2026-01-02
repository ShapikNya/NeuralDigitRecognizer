using System;
using System.IO;
using System.Linq;

class NeuralNetwork
{
    public int[] layers;
    public double[][] neurons;
    public double[][][] weights;
    public double learningRate;

    public NeuralNetwork(int[] layers, double learningRate = 0.1)
    {
        this.layers = layers;
        this.learningRate = learningRate;
        InitializeNetwork();
    }

    void InitializeNetwork()
    {
        Random rand = new Random();
        neurons = new double[layers.Length][];
        weights = new double[layers.Length][][];

        for (int i = 0; i < layers.Length; i++)
            neurons[i] = new double[layers[i]];

        for (int l = 1; l < layers.Length; l++)
        {
            weights[l] = new double[layers[l - 1]][];
            for (int i = 0; i < layers[l - 1]; i++)
            {
                weights[l][i] = new double[layers[l]];
                for (int j = 0; j < layers[l]; j++)
                    weights[l][i][j] = rand.NextDouble() * 0.2 - 0.1;
            }
        }
    }

    static double Sigmoid(double x) => 1.0 / (1.0 + Math.Exp(-x));

    public double[] Forward(double[] input)
    {
        input.CopyTo(neurons[0], 0);
        for (int l = 1; l < layers.Length; l++)
            for (int j = 0; j < layers[l]; j++)
            {
                double sum = 0;
                for (int i = 0; i < layers[l - 1]; i++)
                    sum += neurons[l - 1][i] * weights[l][i][j];
                neurons[l][j] = Sigmoid(sum);
            }
        return neurons[^1];
    }

    public void Backpropagate(double[] y)
    {
        int L = layers.Length - 1;
        double[][] deltas = new double[layers.Length][];
        for (int l = 0; l < layers.Length; l++)
            deltas[l] = new double[layers[l]];

        for (int j = 0; j < layers[L]; j++)
            deltas[L][j] = 2 * (neurons[L][j] - y[j]) * neurons[L][j] * (1 - neurons[L][j]);

        for (int l = L - 1; l > 0; l--)
            for (int i = 0; i < layers[l]; i++)
            {
                double sum = 0;
                for (int j = 0; j < layers[l + 1]; j++)
                    sum += deltas[l + 1][j] * weights[l + 1][i][j];
                deltas[l][i] = sum * neurons[l][i] * (1 - neurons[l][i]);
            }

        for (int l = 1; l < layers.Length; l++)
            for (int i = 0; i < layers[l - 1]; i++)
                for (int j = 0; j < layers[l]; j++)
                    weights[l][i][j] -= learningRate * deltas[l][j] * neurons[l - 1][i];
    }

    public double Error(double[] y)
    {
        double err = 0;
        for (int i = 0; i < y.Length; i++)
            err += Math.Pow(y[i] - neurons[^1][i], 2);
        return err;
    }

    public void SaveWeightsToFile(string path)
    {
        using (StreamWriter sw = new StreamWriter(path))
        {
            for (int l = 1; l < layers.Length; l++)
            {
                for (int i = 0; i < layers[l - 1]; i++)
                {
                    string line = string.Join(",", weights[l][i].Select(w => w.ToString(System.Globalization.CultureInfo.InvariantCulture)));
                    sw.WriteLine(line);
                }
            }
        }
    }

    public static NeuralNetwork LoadFromFile(string path, int[] layers)
    {
        var net = new NeuralNetwork(layers);
        var lines = File.ReadAllLines(path);
        int idx = 0;
        for (int l = 1; l < layers.Length; l++)
            for (int i = 0; i < layers[l - 1]; i++)
            {
                var values = lines[idx++].Split(',').Select(double.Parse).ToArray();
                for (int j = 0; j < layers[l]; j++)
                    net.weights[l][i][j] = values[j];
            }
        return net;
    }
}

class MNISTLoader
{
    public static void LoadCSV(string path, out double[][] inputs, out double[][] labels)
    {
        var lines = File.ReadAllLines(path).Skip(1).ToArray();
        int count = lines.Length;
        inputs = new double[count][];
        labels = new double[count][];
        for (int i = 0; i < count; i++)
        {
            var parts = lines[i].Split(',');
            int label = int.Parse(parts[0]);
            inputs[i] = parts.Skip(1).Select(x => double.Parse(x) / 255.0).ToArray();
            labels[i] = new double[10];
            labels[i][label] = 1.0;
        }
    }
}
class Program
{
    static void Main()
    {
        Console.Write("Enter number of training iterations: ");
        int iterations = int.Parse(Console.ReadLine());

        double[][] trainX, trainY;
        MNISTLoader.LoadCSV("mnist_train.csv", out trainX, out trainY);

        NeuralNetwork nn = new NeuralNetwork(new int[] { 784, 128, 64, 10 }, 0.1);

        Random rnd = new Random();
        int batchSize = 10;
        double batchError = 0;

        for (int iter = 1; iter <= iterations; iter++)
        {
            batchError = 0;
            for (int b = 0; b < batchSize; b++)
            {
                int idx = rnd.Next(trainX.Length);
                nn.Forward(trainX[idx]);
                nn.Backpropagate(trainY[idx]);
                batchError += nn.Error(trainY[idx]);
            }

            if (iter % 10 == 0)
                Console.WriteLine($"Iteration {iter}, Average batch error: {batchError / batchSize:F4}");
        }

        Console.WriteLine("Training finished.");
        Console.WriteLine("Final batch error: " + batchError / batchSize);

        nn.SaveWeightsToFile("trained_weights.txt");
        Console.WriteLine("Weights saved to 'trained_weights.txt'.");
    }
}
