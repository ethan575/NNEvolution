using Jitter2.LinearMath;
using MathNet.Numerics.LinearAlgebra;

namespace _3D_Engine.Classes.NeuralNet
{
    /// <summary>
    /// A fixed set of target schedules that is shared by a whole generation.
    /// Resampling the targets every generation made scores incomparable between
    /// generations, so the best-fitness curve could not distinguish "no progress"
    /// from "a different, harder task". One bank is pinned for <see cref="GenerationsPerBank"/>
    /// generations, then rotated.
    /// </summary>
    public sealed class TargetScheduleBank
    {
        public const int GenerationsPerBank = 10;
        public const int SchedulesPerBank = 3;

        public int BankId { get; }
        public JVector[][] Schedules { get; }

        public TargetScheduleBank(int bankId, JVector[][] schedules)
        {
            BankId = bankId;
            Schedules = schedules;
        }
    }

    public class Generation
    {
        public float ActiveTimeLeft = 20f;
        public List<NeuralNetwork> Population = new List<NeuralNetwork>();

        public Generation()
        {

        }
        public Generation(List<NeuralNetwork> population)
        {
            Population = population;
        }

        public Generation Copy()
        {
            List<NeuralNetwork> newPopulation = new List<NeuralNetwork>();
            foreach (var nn in Population)
            {
                NeuralNetwork copy = nn.Copy();
                copy.FitnessScore = nn.FitnessScore; // keep history scores inspectable
                newPopulation.Add(copy);
            }

            Generation copyGen = new Generation(newPopulation);
            copyGen.ActiveTimeLeft = ActiveTimeLeft;
            return copyGen;
        }
    }

    // contains all the weights and biases of a neural network
    // this is evolved and mutated to create new data for the next generation of neural networks
    public class Genome
    {
        public List<Matrix<float>> Weights;
        public List<Matrix<float>> Biases;
        public Genome(List<Matrix<float>> weights, List<Matrix<float>> biases)
        {
            this.Weights = weights;
            this.Biases = biases;
        }

        public Genome Copy()
        {
            List<Matrix<float>> newWeights = new List<Matrix<float>>();
            List<Matrix<float>> newBiases = new List<Matrix<float>>();
            foreach (var weight in Weights)
            {
                newWeights.Add(weight.Clone());
            }
            foreach (var bias in Biases)
            {
                newBiases.Add(bias.Clone());
            }
            return new Genome(newWeights, newBiases);
        }
    }

    public static class NNTrainer
    {

        // Elites Unmodified top performers

        //Crossover
        //Combine two parents

        //Mutated offspring
        //Copy and mutate a selected parent

        //Immigrants
        //Completely new random networks

        static Random rng = new Random();
        public static List<Generation> Generations = new List<Generation>();
        public static Generation? LatestGen;

        public static TargetScheduleBank? CurrentBank { get; private set; }
        private static int bankGenerationCount = 0;
        private static int nextBankId = 0;

        /// <summary>
        /// Returns the schedule bank to evaluate the next generation on, rotating it every
        /// <see cref="TargetScheduleBank.GenerationsPerBank"/> generations. The scene
        /// supplies the generator because it owns the target geometry and spawn point.
        /// </summary>
        public static TargetScheduleBank NextScheduleBank(TargetScheduleBank? current, Func<JVector[][]> generator)
        {
            if (current == null || bankGenerationCount >= TargetScheduleBank.GenerationsPerBank)
            {
                bankGenerationCount = 0;
                CurrentBank = new TargetScheduleBank(nextBankId++, generator());
                return CurrentBank;
            }

            CurrentBank = current;
            return current;
        }

        public static void CountEvaluatedGeneration()
        {
            bankGenerationCount++;
        }

        private static List<NeuralNetwork> EliteNNs = new List<NeuralNetwork>();
        private static List<NeuralNetwork> NewNNs = new List<NeuralNetwork>();

        private const float DefaultMutationRate = 0.2f;
        private const int MaxGenerationsKept = 50;

        public static Generation EvolveGeneration(List<NeuralNetwork> population, int NoEliteNNs = 8)
        {
            if (population == null || population.Count == 0)
            {
                var emptyGen = new Generation(new List<NeuralNetwork>());
                LatestGen = emptyGen;
                return emptyGen;
            }

            List<NeuralNetwork> sorted = population.OrderByDescending(nn => nn.FitnessScore).ToList();
            int eliteCount = Math.Clamp(NoEliteNNs, 1, sorted.Count);
            int offspringCount = sorted.Count - eliteCount;

            EliteNNs = sorted.Take(eliteCount).Select(nn => 
            {var copy = nn.Copy();  
                copy.ExecuteEndOfGenerationTasks();
                return copy;
            }).ToList();

            NewNNs = new List<NeuralNetwork>(offspringCount);

            if (offspringCount > 0)
            {
                // Parents are drawn from a pool that EXCLUDES the elites. Elites are already
                // copied verbatim into the next generation, so letting them back into the
                // breeding pool means roughly 30 of 50 slots were near-duplicates of the top
                // 8 and effective diversity was far lower than the population size suggested.
                int poolSize = sorted.Count - eliteCount;
                int immigrantCount = offspringCount >= 10 ? 5 : 0;
                int tunableCount = offspringCount - immigrantCount;
                int crossoverCount = poolSize >= 2 ? Math.Min(Math.Max(0, tunableCount / 6), tunableCount) : 0;

                for (int i = 0; i < offspringCount; i++)
                {
                    if (i >= tunableCount)
                    {
                        // Fresh random immigrants preserve global exploration
                        var immigrant = new NeuralNetwork(sorted[0].LayerDims);
                        immigrant.ExecuteEndOfGenerationTasks();
                        NewNNs.Add(immigrant);
                        continue;
                    }

                    NeuralNetwork child;
                    if (i < crossoverCount && poolSize >= 2)
                    {
                        int p1 = SelectTournamentIndex(eliteCount, poolSize, tournamentSize: 3);
                        int p2;
                        do
                        {
                            p2 = SelectTournamentIndex(eliteCount, poolSize, tournamentSize: 3);
                        } while (p2 == p1 && poolSize > 1);

                        child = NeuronCrossover(sorted[p1], sorted[p2]);
                        MutateNetwork(child, DefaultMutationRate, sigma: 0.15f);
                        
                    }
                    else
                    {
                        int parentIdx = SelectTournamentIndex(eliteCount, poolSize, tournamentSize: 3);
                        child = sorted[parentIdx].Copy();

                        int mutIdx = i - crossoverCount;
                        int mutSpan = Math.Max(1, tunableCount - crossoverCount);
                        float frac = (float)mutIdx / mutSpan;

                        // Roughly 400 weights are edited per child at rate 0.2, so the old
                        // 0.18/0.35 sigmas were very large steps relative to the noise in a
                        // per-generation fitness. Smaller steps on the fitter half of the
                        // offspring, with a few genuinely exploratory children at the tail.
                        if (frac < 0.35f)
                            MutateNetwork(child, rate: 0.15f, sigma: 0.05f);
                        else if (frac < 0.80f)
                            MutateNetwork(child, rate: 0.20f, sigma: 0.12f);
                        else
                            MutateNetwork(child, rate: 0.30f, sigma: 0.30f);
                    }

                    child.ExecuteEndOfGenerationTasks();
                    NewNNs.Add(child);
                }
            }

            List<NeuralNetwork> nextPopulation = new List<NeuralNetwork>(sorted.Count);
            nextPopulation.AddRange(EliteNNs);
            nextPopulation.AddRange(NewNNs);

            Generation newGen = new Generation(nextPopulation);
            population.ForEach(nn => nn.FitnessScore = 0); // reset old fitness scores

            // Keep the history: Generations was declared but never written to, so there was
            // no way to inspect or roll back to an earlier population.
            Generations.Add(newGen.Copy());
            if (Generations.Count > MaxGenerationsKept)
            {
                Generations.RemoveAt(0);
            }

            LatestGen = newGen;
            CountEvaluatedGeneration();
            return newGen;
        }

        public static NeuralNetwork CreateMutatedVariant(NeuralNetwork parent, float rate = DefaultMutationRate, float stDev = 0.18f)
        {
            NeuralNetwork variant = parent.Copy();
            MutateNetwork(variant, rate, stDev);
            variant.ExecuteEndOfGenerationTasks();
            return variant;
        }

        // Select an index into `sorted`, looking only at [offset, offset + poolSize), and
        // biased toward earlier (fitter) entries.
        private static int SelectTournamentIndex(int offset, int poolSize, int tournamentSize)
        {
            int best = offset + rng.Next(poolSize);
            for (int t = 1; t < tournamentSize; t++)
            {
                int candidate = offset + rng.Next(poolSize);
                if (candidate < best)
                    best = candidate;
            }
            return best;
        }

        // Crosses entire neuron rows (incoming weights + that neuron's bias) together so hidden units stay coherent
        private static NeuralNetwork NeuronCrossover(NeuralNetwork parent1, NeuralNetwork parent2)
        {
            NeuralNetwork child = parent1.Copy();

            for (int layerWeightIdx = 0; layerWeightIdx < child.NetworkWeights.Count; layerWeightIdx++)
            {
                Matrix<float> childWeights = child.NetworkWeights[layerWeightIdx];
                Matrix<float> p1Weights = parent1.NetworkWeights[layerWeightIdx];
                Matrix<float> p2Weights = parent2.NetworkWeights[layerWeightIdx];

                int targetLayerIdx = layerWeightIdx + 1;
                bool hasTargetBias = targetLayerIdx < child.NetworkLayers.Count;

                for (int row = 0; row < childWeights.RowCount; row++)
                {
                    bool pickFirst = rng.NextSingle() < 0.5f;
                    Matrix<float> srcWeights = pickFirst ? p1Weights : p2Weights;

                    for (int col = 0; col < childWeights.ColumnCount; col++)
                    {
                        childWeights[row, col] = srcWeights[row, col];
                    }

                    if (hasTargetBias && row < child.NetworkLayers[targetLayerIdx].LayerBias.RowCount)
                    {
                        Matrix<float> srcBias = pickFirst
                            ? parent1.NetworkLayers[targetLayerIdx].LayerBias
                            : parent2.NetworkLayers[targetLayerIdx].LayerBias;
                        child.NetworkLayers[targetLayerIdx].LayerBias[row, 0] = srcBias[row, 0];
                    }
                }
            }

            return child;
        }

        private static void MutateNetwork(NeuralNetwork network, float rate, float sigma)
        {
            for (int layerWeightIdx = 0; layerWeightIdx < network.NetworkWeights.Count; layerWeightIdx++)
            {
                Mutate(network.NetworkWeights[layerWeightIdx], rate, sigma);
            }

            // Skip layer 0 (InputLayer bias is unused)
            for (int layerBiasIdx = 1; layerBiasIdx < network.NetworkLayers.Count; layerBiasIdx++)
            {
                Mutate(network.NetworkLayers[layerBiasIdx].LayerBias, rate, sigma);
            }
        }

        private static void Mutate(Matrix<float> matrix, float rate, float sigma)
        {
            for (int row = 0; row < matrix.RowCount; row++)
            {
                for (int column = 0; column < matrix.ColumnCount; column++)
                {
                    if (rng.NextSingle() < rate)
                        matrix[row, column] += (float)gaussian(0.0, sigma);
                }
            }
        }

        // https://github.com/conradshyu/rng/blob/master/random.cs
        static double gaussian(double mean = 0.0, double s = 1.0)
        {
            double u1;
            do u1 = uniform(); while (u1 <= 0.0);   // remove 0.0 to avoid log(0)
            double u2 = uniform();

            return s * Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Cos(2.0 * Math.PI * u2) + mean;
        }

        static double uniform(double a = 0.0, double b = 1.0)
        {
            return a + (b - a) * rng.NextDouble();
        }

    }
}
