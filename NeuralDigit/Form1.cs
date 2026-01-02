using System;
using System.Drawing;
using System.Linq;
using System.Windows.Forms;
using System.IO;

namespace NeuralDigit
{
    public partial class Form1 : Form
    {
        PictureBox canvas;
        Button btnRecognize, btnClear;
        Bitmap bitmap;
        Graphics graphics;
        NeuralNetwork network;

        public Form1()
        {
            InitializeComponent();
            BuildUI();
            InitDrawing();
            network = NeuralNetwork.LoadFromFile("trained_weights.txt", new int[] { 784, 128, 64, 10 });
        }

        private void BuildUI()
        {
            this.Text = "Neural Digit Recognizer";
            this.ClientSize = new Size(320, 360);

            canvas = new PictureBox
            {
                Location = new Point(20, 20),
                Size = new Size(280, 280),
                BorderStyle = BorderStyle.FixedSingle
            };
            btnRecognize = new Button { Text = "Recognize", Location = new Point(20, 310), Width = 120 };
            btnClear = new Button { Text = "Clear", Location = new Point(160, 310), Width = 120 };

            Controls.Add(canvas);
            Controls.Add(btnRecognize);
            Controls.Add(btnClear);

            canvas.MouseMove += Canvas_MouseMove;
            btnRecognize.Click += BtnRecognize_Click;
            btnClear.Click += BtnClear_Click;
        }

        private void InitDrawing()
        {
            bitmap = new Bitmap(28, 28);
            graphics = Graphics.FromImage(bitmap);
            graphics.Clear(Color.Black);
            canvas.Image = new Bitmap(bitmap, 280, 280);
        }

        private void Canvas_MouseMove(object sender, MouseEventArgs e)
        {
            if (e.Button != MouseButtons.Left) return;
            int x = e.X * 28 / canvas.Width;
            int y = e.Y * 28 / canvas.Height;
            if (x >= 0 && x < 28 && y >= 0 && y < 28)
            {
                bitmap.SetPixel(x, y, Color.White);
                canvas.Image = new Bitmap(bitmap, 280, 280);
            }
        }

        private void BtnClear_Click(object sender, EventArgs e)
        {
            graphics.Clear(Color.Black);
            canvas.Image = new Bitmap(bitmap, 280, 280);
        }

        private void BtnRecognize_Click(object sender, EventArgs e)
        {
            double[] input = new double[784];
            for (int y = 0; y < 28; y++)
                for (int x = 0; x < 28; x++)
                    input[y * 28 + x] = bitmap.GetPixel(x, y).R / 255.0;

            double[] output = network.Forward(input);
            int digit = Array.IndexOf(output, output.Max());
            MessageBox.Show($"Распознанная цифра: {digit}", "Result");
        }
    }
    public class NeuralNetwork
    {
        int[] layers;
        double[][] neurons;
        double[][][] weights;

        public NeuralNetwork(int[] layers)
        {
            this.layers = layers;
            neurons = new double[layers.Length][];
            weights = new double[layers.Length][][];

            for (int l = 0; l < layers.Length; l++)
                neurons[l] = new double[layers[l]];

            for (int l = 1; l < layers.Length; l++)
            {
                weights[l] = new double[layers[l - 1]][];
                for (int i = 0; i < layers[l - 1]; i++)
                    weights[l][i] = new double[layers[l]];
            }
        }

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

        double Sigmoid(double x) => 1.0 / (1.0 + Math.Exp(-x));

        public static NeuralNetwork LoadFromFile(string path, int[] layers)
        {
            var net = new NeuralNetwork(layers);
            var lines = File.ReadAllLines(path);
            int idx = 0;
            for (int l = 1; l < layers.Length; l++)
                for (int i = 0; i < layers[l - 1]; i++)
                {
                    var values = lines[idx++]
                        .Split(',')
                        .Select(s => double.Parse(s, System.Globalization.CultureInfo.InvariantCulture))
                        .ToArray();
                    for (int j = 0; j < layers[l]; j++)
                        net.weights[l][i][j] = values[j];
                }
            return net;
        }
    }
}
