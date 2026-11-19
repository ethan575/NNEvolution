using _3D_Engine.Classes.Ragdoll;
using Jitter2.LinearMath;
using MathNet.Numerics.Statistics;

namespace _3D_Engine.Classes.NeuralNet
{

    public class FitnessObjective
    {
        public float weight;
        public Func<RagdollBody, float> fitnessFunction;

        public FitnessObjective(float weight, Func<RagdollBody, float> fitnessFunction)
        {
            this.weight = weight;
            this.fitnessFunction = fitnessFunction;
        }
    }

    public static class Objectives
    {
        public static float EvalProgressToTarget(JVector initialPosition, JVector finalPosition, JVector targetPosition)
        {
            float startDx = targetPosition.X - initialPosition.X;
            float startDz = targetPosition.Z - initialPosition.Z;
            float startDistance = MathF.Sqrt(startDx * startDx + startDz * startDz);

            float finalDx = targetPosition.X - finalPosition.X;
            float finalDz = targetPosition.Z - finalPosition.Z;
            float finalDistance = MathF.Sqrt(finalDx * finalDx + finalDz * finalDz);

            if (startDistance < 0.001f)
                return 0f;

            float progress = (startDistance - finalDistance) / startDistance;

            const float arrivalRadius = 10f;

            float arrivalBonus = finalDistance <= arrivalRadius ? 0.6f : 0f;

            return progress + arrivalBonus;
        }

        public static float EvalMultiPhaseTargetFollowing(RagdollBody ragdoll)
        {
            // NOTE: progress must be scored *relative to the phase's starting distance*.
            // Raw metres are dominated by walking speed: a creature that merely drifts fast
            // in one fixed direction used to out-score one that actually steers towards the
            // target. Relative progress is bounded to [-1, +1] per phase, so speed alone
            // can never earn anything and "arrived" always beats "moved a long way".
            if (ragdoll.PhaseProgressMeters.Count == 0 || ragdoll.PhaseStartDistances.Count == 0)
            {
                // Same scale as the per-phase path (at most TargetPhasesPerGeneration units),
                // so an interrupted generation cannot score ~20x its real value.
                float relative = EvalProgressToTarget(ragdoll.initialPos, ragdoll.COMPosition, ragdoll.TargetPosition);
                return 3f * relative;
            }

            float total = 0f;
            int count = Math.Min(ragdoll.PhaseProgressMeters.Count, ragdoll.PhaseStartDistances.Count);

            for (int i = 0; i < count; i++)
            {
                float startDistance = ragdoll.PhaseStartDistances[i];
                if (startDistance < 1e-3f)
                {
                    continue;
                }

                // -1 = walked directly away from the target, +1 = arrived at it.
                total += Math.Clamp(ragdoll.PhaseProgressMeters[i] / startDistance, -1f, 1f);
            }

            // Arrival must dominate everything else. Reaching a target also implies
            // roughly +1 relative progress, so one arrival is worth about as much as
            // four phases of perfect-but-unfinished tracking.
            return total + 4f * ragdoll.PhasesReachedCount;
        }

        public static float EvalEnergyUsed(float EnergyUsed, int stepsTaken)
        {
            if (stepsTaken <= 0)
                return 0;

            // EnergyUsed is a sum over every control step, so it has to be normalised by
            // the number of steps or the score is identically ~0 for any real creature.
            float meanEffort = EnergyUsed / stepsTaken;
            return 1f / (1f + meanEffort);
        }

        public static float EvalTimeTaken(float TimeTaken, bool reached)
        {
            if (TimeTaken <= 0 || !reached)
                return 0;

            // ~1.0 for an instant arrival, ~0.25 for one that takes the whole phase.
            return 1f / (1f + TimeTaken);
        }

    }

    public static class FitnessEval
    {
        public static float EvaluateFitness(RagdollBody ragdoll)
        {
            float totalFitness = 0f;

            foreach (FitnessObjective objective in ragdoll.fitnessObjectives)
            {
                totalFitness += objective.weight * objective.fitnessFunction(ragdoll);
            }
            return totalFitness;
        }
    }
}
