using _3D_Engine.Classes.NeuralNet;
using Jitter2;
using Jitter2.Dynamics;
using Jitter2.LinearMath;
using MathNet.Numerics.LinearAlgebra;
using Vector3 = OpenTK.Mathematics.Vector3;

namespace _3D_Engine.Classes.Ragdoll
{
    public struct RagdollSnapshot
    {

        public JVector COMPosition { get; set; }
        public JVector[] BonePositions { get; set; }
        public JVector[] BoneAngularVelocities { get; set; }
        public JVector[] BoneVelocities { get; set; }
        public JVector BodyVelocity { get; set; }
        public JQuaternion[] BoneRotations { get; set; }

        public RagdollSnapshot(JVector comPosition, JVector bodyVelocity, JVector[] boneVelocities, JQuaternion[] boneRotations)
        {

            BonePositions = Array.Empty<JVector>();
            BoneAngularVelocities = Array.Empty<JVector>();
            COMPosition = comPosition;
            BodyVelocity = bodyVelocity;
            BoneVelocities = boneVelocities;
            BoneRotations = boneRotations;

        }
    }

    public class RagdollBody
    {
        public const float MaxJointMotorVelocity = 10f; // rad/s
        public const float ArrivalRadius = 10f;

        public List<Bone> Bones { get; set; }
        public List<Joint> Joints { get; set; }

        public NeuralNetwork Brain { get; private set; }

        public JVector BodyVelocity { get; private set; }

        public JVector TargetPosition = JVector.Zero;
        public JVector COMPosition { get; private set; } = JVector.Zero;

        public float EnergyUsed = 0;
        public int ControlSteps = 0;
        public float TimeToReachTarget = 0;
        public bool reached = false;

        public JVector initialPos;

        public List<float> PhaseProgressMeters { get; } = new List<float>();
        // Distance to the phase target when the phase started. Progress is scored relative
        // to this, which makes the reward speed-independent.
        public List<float> PhaseStartDistances { get; } = new List<float>();
        public List<float> PhaseTimesToReach { get; } = new List<float>();
        public int PhasesReachedCount { get; private set; } = 0;

        private JVector phaseStartPos = JVector.Zero;
        private JVector phaseTargetPos = JVector.Zero;
        private bool hasActivePhase = false;
        private float phaseMinHorizontalDist = float.MaxValue;
        private bool phaseReached = false;
        private float phaseClock = 0f;
        private float phaseStartClock = 0f;
        private float phaseStartDist = 0f;

        private float period = 1.5f; // period of the sine wave for the clock input

        public RagdollSnapshot initialConfig { get; private set; }

        public FitnessObjective[] fitnessObjectives = new FitnessObjective[]
        {
            new FitnessObjective(1f, ragdoll => Objectives.EvalMultiPhaseTargetFollowing(ragdoll)),
            new FitnessObjective(0f, ragdoll => Objectives.EvalEnergyUsed(ragdoll.EnergyUsed, ragdoll.ControlSteps)),
            new FitnessObjective(0.5f, ragdoll => Objectives.EvalTimeTaken(ragdoll.TimeToReachTarget, ragdoll.reached))
        };

        private float clock = 0f;

        public int RequiredInputCount => (this.Joints.Count * 2) + 12;

        public RagdollBody(List<Bone> bones, List<Joint> joints)
        {
            this.Bones = bones ?? new List<Bone>();
            this.Joints = joints ?? new List<Joint>();

            COMPosition = ComputeCentreOfMass();
            BodyVelocity = ComputeVelocity();
            initialPos = COMPosition;
            this.initialConfig = new RagdollSnapshot
            {
                COMPosition = COMPosition,
                BodyVelocity = BodyVelocity,
                BonePositions = Bones.Select(b => b.RB.Position).ToArray(),
                BoneAngularVelocities = Bones.Select(b => b.RB.AngularVelocity).ToArray(),
                BoneVelocities = Bones.Select(b => b.RB.Velocity).ToArray(),
                BoneRotations = Bones.Select(b => b.RB.Orientation).ToArray()
            };
        }

        public void SetBrain(NeuralNetwork brain)
        {
            /*
             inputs = [
                each joint angle (normalized to +/- 45 deg limit),
                each joint angular velocity,
                COM velocity in reference bone local frame (3),
                direction to target in reference bone local frame (3),
                reference bone local up vector (3),
                normalized distance to target (1),
                sin(clock),
                cos(clock)
             ]
             */

            if (brain.LayerDims[0] != RequiredInputCount)
            {
                throw new ArgumentException($"Input layer size ({brain.LayerDims[0]}) must equal twice the joint count plus twelve ({RequiredInputCount}).");
            }
            if (brain.LayerDims[brain.LayerDims.Length - 1] != this.Joints.Count)
            {
                throw new ArgumentException($"Output layer size ({brain.LayerDims[brain.LayerDims.Length - 1]}) must equal joint count ({this.Joints.Count}).");
            }
            Brain = brain;
        }

        public JVector ComputeCentreOfMass()
        {
            JVector centerOfMass = JVector.Zero;
            float totalMass = 0f;
            foreach (var bone in Bones)
            {
                float mass = bone.RB.Mass;
                centerOfMass += bone.RB.Position * mass;
                totalMass += mass;
            }
            if (totalMass > 0)
            {
                centerOfMass = centerOfMass * (1 / totalMass);
            }
            return centerOfMass;
        }

        public JVector ComputeVelocity()
        {
            JVector total = JVector.Zero;
            float totalMass = 0;

            foreach (Bone b in this.Bones)
            {
                total += b.RB.Mass * b.RB.Velocity;
                totalMass += b.RB.Mass;
            }

            return totalMass > 0 ? total * (1 / totalMass) : JVector.Zero;
        }

        public static float HorizontalDistance(JVector a, JVector b)
        {
            float dx = a.X - b.X;
            float dz = a.Z - b.Z;
            return MathF.Sqrt(dx * dx + dz * dz);
        }

        public void StartTargetPhase(JVector newTargetPosition)
        {
            if (hasActivePhase)
            {
                FinalizeCurrentPhase();
            }

            TargetPosition = newTargetPosition;
            phaseTargetPos = newTargetPosition;
            COMPosition = ComputeCentreOfMass();
            phaseStartPos = COMPosition;
            phaseMinHorizontalDist = HorizontalDistance(phaseStartPos, phaseTargetPos);

            // Per-phase bookkeeping: arrival and timing have to restart with each phase,
            // otherwise "time to reach target" is cumulative and only ever describes phase 1.
            phaseReached = false;
            phaseClock = 0f;
            phaseStartClock = clock;
            phaseStartDist = phaseMinHorizontalDist;

            hasActivePhase = true;
        }

        public void FinalizeCurrentPhase()
        {
            if (!hasActivePhase) return;

            COMPosition = ComputeCentreOfMass();
            float startDist = HorizontalDistance(phaseStartPos, phaseTargetPos);
            float endDist = HorizontalDistance(COMPosition, phaseTargetPos);

            // Credit the *closest approach* rather than the end-of-phase position. A
            // creature that reaches the target and then carries on past it has succeeded at
            // "get to the target", and scoring the end position instead punished any
            // creature that overshoots - i.e. it punished exactly the speed we want.
            float bestDist = MathF.Min(endDist, phaseMinHorizontalDist);
            float progressMeters = startDist - bestDist;

            PhaseProgressMeters.Add(progressMeters);
            PhaseStartDistances.Add(startDist);

            bool phaseArrived = bestDist <= ArrivalRadius;
            if (phaseArrived)
            {
                PhasesReachedCount++;
                phaseReached = true;
                // Time to reach the *first* target of the generation, measured in seconds.
                if (!reached)
                {
                    TimeToReachTarget = phaseClock;
                    reached = true;
                }
                PhaseTimesToReach.Add(phaseClock);
            }
            else
            {
                PhaseTimesToReach.Add(0f);
            }

            hasActivePhase = false;
        }

        public Matrix<float> GetBrainInputMatrix()
        {
            if (Brain == null)
            {
                throw new InvalidOperationException("Brain is not set.");
            }

            // Inputs must reflect physics even if no render frame has run.
            COMPosition = ComputeCentreOfMass();
            BodyVelocity = ComputeVelocity();
            Matrix<float> inputMatrix = Matrix<float>.Build.Dense(Brain.LayerDims[0], 1);

            // A dedicated body frame, not Bones[0]. Bones[0] is whichever bone happened to
            // be first in the .creature file, so the meaning of "forward" flipped with its
            // roll and could change between saves. This frame is aligned to the "head"
            // (highest bone above the COM) with world up as a stable secondary axis, which
            // keeps the yaw sign consistent for every creature.
            JQuaternion invRefOrientation = JQuaternion.Conjugate(ComputeBodyFrameOrientation());

            // Joint angles & angular velocities
            for (int i = 0; i < Joints.Count; i++)
            {
                Joint joint = Joints[i];
                float angleNorm = joint.HindgeJointConstraint.HingeAngle.Angle.Radian / (MathF.PI / 4f);
                float angVelNorm = joint.AngularVelocityRad() / 8f;

                inputMatrix[2 * i, 0] = float.IsFinite(angleNorm) ? Math.Clamp(angleNorm, -2f, 2f) : 0f;
                inputMatrix[2 * i + 1, 0] = float.IsFinite(angVelNorm) ? Math.Clamp(angVelNorm, -2f, 2f) : 0f;
            }

            int lastWrittenIdx = 2 * Joints.Count;

            // Body velocity in reference bone's local frame
            JVector localVel = JVector.Transform(this.BodyVelocity, invRefOrientation);
            inputMatrix[lastWrittenIdx + 0, 0] = float.IsFinite(localVel.X) ? Math.Clamp(localVel.X / 10f, -2f, 2f) : 0f;
            inputMatrix[lastWrittenIdx + 1, 0] = float.IsFinite(localVel.Y) ? Math.Clamp(localVel.Y / 10f, -2f, 2f) : 0f;
            inputMatrix[lastWrittenIdx + 2, 0] = float.IsFinite(localVel.Z) ? Math.Clamp(localVel.Z / 10f, -2f, 2f) : 0f;

            // Direction to target in reference bone's local frame
            JVector worldDir = TargetPosition - COMPosition;
            float dist = worldDir.Length();
            JVector worldUnitDir = dist > 1e-4f ? worldDir * (1f / dist) : JVector.Zero;
            JVector localDir = JVector.Transform(worldUnitDir, invRefOrientation);
            inputMatrix[lastWrittenIdx + 3, 0] = float.IsFinite(localDir.X) ? localDir.X : 0f;
            inputMatrix[lastWrittenIdx + 4, 0] = float.IsFinite(localDir.Y) ? localDir.Y : 0f;
            inputMatrix[lastWrittenIdx + 5, 0] = float.IsFinite(localDir.Z) ? localDir.Z : 0f;

            // World up vector in reference bone's local frame (tilt / inversion awareness)
            JVector localUp = JVector.Transform(JVector.UnitY, invRefOrientation);
            inputMatrix[lastWrittenIdx + 6, 0] = float.IsFinite(localUp.X) ? localUp.X : 0f;
            inputMatrix[lastWrittenIdx + 7, 0] = float.IsFinite(localUp.Y) ? localUp.Y : 1f;
            inputMatrix[lastWrittenIdx + 8, 0] = float.IsFinite(localUp.Z) ? localUp.Z : 0f;

            // Distance to target, normalised by the phase's starting distance.
            // This used to be a fixed /30f divisor, but the targets sat 32-46 units away so
            // the input was pinned at 1.0 for the whole generation and carried nothing.
            float distanceReference = phaseStartDist > 1f ? phaseStartDist : 50f;
            inputMatrix[lastWrittenIdx + 9, 0] = float.IsFinite(dist)
                ? Math.Clamp(dist / distanceReference, 0f, 1f)
                : 1f;

            // Sin / cos periodic clock
            inputMatrix[lastWrittenIdx + 10, 0] = MathF.Sin(2f * MathF.PI * clock / period);
            inputMatrix[lastWrittenIdx + 11, 0] = MathF.Cos(2f * MathF.PI * clock / period);

            return inputMatrix;
        }

        public void BrainNNFF(Matrix<float> input)
        {
            if (Brain == null)
            {
                throw new InvalidOperationException("Brain is not set.");
            }
            else if (input.RowCount != Brain.LayerDims[0] || input.ColumnCount != 1)
            {
                throw new ArgumentException("Input matrix dimensions do not match the expected input layer size.");
            }

            COMPosition = ComputeCentreOfMass();
            float horizontalDist = HorizontalDistance(COMPosition, TargetPosition);
            ControlSteps++;
            if (hasActivePhase)
            {
                phaseMinHorizontalDist = MathF.Min(phaseMinHorizontalDist, horizontalDist);

                // Per-phase arrival is recorded in FinalizeCurrentPhase so that every phase
                // reports its own time (and only counts as reached if it really was).
                if (!phaseReached && horizontalDist <= ArrivalRadius)
                {
                    phaseReached = true;
                    phaseClock = clock - phaseStartClock;
                }
            }

            // Keep the brain active after arriving so it continues balancing and following.
            if (!reached && horizontalDist <= ArrivalRadius)
            {
                TimeToReachTarget = clock - phaseStartClock;
                reached = true;
            }

            Matrix<float> output = Brain.GetNetworkOutput(input);

            for (int i = 0; i < Joints.Count; i++)
            {
                float o = output[i, 0];
                if (float.IsNaN(o) || float.IsInfinity(o)) o = 0f;
                float t = Math.Clamp(o * MaxJointMotorVelocity, -MaxJointMotorVelocity, MaxJointMotorVelocity);

                Joints[i].HindgeJointConstraint.Motor.TargetVelocity = t;
                EnergyUsed += MathF.Abs(o);
            }
        }

        /// <summary>
        /// Builds a stable locomotion frame for the proprioceptive inputs.
        /// Yaw is taken from the "head" (the bone furthest above the centre of mass), pitch is
        /// derived from world up, and roll is zeroed so the frame cannot spin around the
        /// head-tail axis. Returns identity when the creature has no vertical extent.
        /// </summary>
        private JQuaternion ComputeBodyFrameOrientation()
        {
            if (Bones == null || Bones.Count == 0)
            {
                return JQuaternion.Identity;
            }

            // COM is refreshed by the caller immediately before this is used.
            int headIdx = 0;
            float bestHeight = float.MinValue;
            for (int i = 0; i < Bones.Count; i++)
            {
                float dy = Bones[i].RB.Position.Y - COMPosition.Y;
                if (dy > bestHeight)
                {
                    bestHeight = dy;
                    headIdx = i;
                }
            }

            if (bestHeight < 0.05f)
            {
                return JQuaternion.Identity;
            }

            JVector up = JVector.UnitY;
            JVector forward = Bones[headIdx].RB.Position - COMPosition;
            forward.Y = 0f;

            if (forward.LengthSquared() < 1e-6f || !IsFinite(forward))
            {
                // Creature is standing upright, so its heading is undefined: fall back to
                // the head bone's own forward axis, flattened.
                forward = JVector.Transform(JVector.UnitX, Bones[headIdx].RB.Orientation);
                forward.Y = 0f;
            }

            if (forward.LengthSquared() < 1e-6f || !IsFinite(forward))
            {
                return JQuaternion.Identity;
            }

            JVector.NormalizeInPlace(ref forward);
            JVector right = JVector.Cross(up, forward);
            if (right.LengthSquared() < 1e-6f || !IsFinite(right))
            {
                return JQuaternion.Identity;
            }
            JVector.NormalizeInPlace(ref right);

            return QuaternionFromAxes(right, up, forward);
        }

        private static bool IsFinite(JVector v)
        {
            return float.IsFinite(v.X) && float.IsFinite(v.Y) && float.IsFinite(v.Z);
        }

        /// <summary>
        /// Converts an orthonormal basis whose columns are (right, up, forward) into a
        /// quaternion. Written as an explicit matrix so it cannot depend on the axis
        /// convention of whichever maths library is in play.
        /// </summary>
        private static JQuaternion QuaternionFromAxes(JVector right, JVector up, JVector forward)
        {
            float m00 = right.X, m01 = up.X, m02 = forward.X;
            float m10 = right.Y, m11 = up.Y, m12 = forward.Y;
            float m20 = right.Z, m21 = up.Z, m22 = forward.Z;

            float trace = m00 + m11 + m22;
            float x, y, z, w;

            if (trace > 0f)
            {
                float s = MathF.Sqrt(trace + 1f) * 2f;
                w = 0.25f * s;
                x = (m21 - m12) / s;
                y = (m02 - m20) / s;
                z = (m10 - m01) / s;
            }
            else if (m00 > m11 && m00 > m22)
            {
                float s = MathF.Sqrt(1f + m00 - m11 - m22) * 2f;
                w = (m21 - m12) / s;
                x = 0.25f * s;
                y = (m01 + m10) / s;
                z = (m02 + m20) / s;
            }
            else if (m11 > m22)
            {
                float s = MathF.Sqrt(1f + m11 - m00 - m22) * 2f;
                w = (m02 - m20) / s;
                x = (m01 + m10) / s;
                y = 0.25f * s;
                z = (m12 + m21) / s;
            }
            else
            {
                float s = MathF.Sqrt(1f + m22 - m00 - m11) * 2f;
                w = (m10 - m01) / s;
                x = (m02 + m20) / s;
                y = (m12 + m21) / s;
                z = 0.25f * s;
            }

            float len = MathF.Sqrt(x * x + y * y + z * z + w * w);
            if (len < 1e-6f)
            {
                return JQuaternion.Identity;
            }

            return new JQuaternion(x / len, y / len, z / len, w / len);
        }

        public void AddBone(Bone bone)
        {
            Bones.Add(bone);
        }
        public void AddJoint(Joint joint)
        {
            Joints.Add(joint);
        }

        public void ApplyTorque(Vector3 torque, Bone bone)
        {
            bone.RB.AddForce(new JVector(torque.X, torque.Y, torque.Z));
        }

        public void ApplyForce(JVector force)
        {
            foreach (Bone b in Bones)
            {
                b.RB.AddForce(force);
            }
        }

        public void ResetAngularVelocities()
        {
            foreach (var item in Joints)
            {
                item.HindgeJointConstraint.Motor.TargetVelocity = 0;
            }
        }

        public void AdvanceSimulationClock(float dt)
        {
            clock += dt;
        }

        public void Update(float dt)
        {
            foreach (var bone in Bones)
            {
                bone.UpdateVisual();
            }
            foreach (var joint in Joints)
            {
                joint.UpdateVisual();
            }

            COMPosition = this.ComputeCentreOfMass();
            this.BodyVelocity = ComputeVelocity();
        }

        public void setPostion(JVector position)
        {
            JVector offset = position - this.ComputeCentreOfMass();
            foreach (var bone in Bones)
            {
                bone.RB.Position += offset;
                bone.UpdateVisual();
            }
        }

        public RagdollBody Clone(World world)
        {
            List<Bone> clonedBones = Bones.Select(b => (Bone)b.Clone(world)).ToList();
            var byId = clonedBones.ToDictionary(b => b.BoneID);

            List<Joint> clonedJoints = Joints.Select(j => j.Clone(world, byId[j.BoneA.BoneID], byId[j.BoneB.BoneID])).ToList();

            RagdollBody clonedRagdoll = new RagdollBody(clonedBones, clonedJoints);
            // Snapshot arrays must not be shared between creatures.
            clonedRagdoll.initialConfig = new RagdollSnapshot
            {
                COMPosition = initialConfig.COMPosition,
                BodyVelocity = initialConfig.BodyVelocity,
                BonePositions = initialConfig.BonePositions.ToArray(),
                BoneAngularVelocities = initialConfig.BoneAngularVelocities.ToArray(),
                BoneVelocities = initialConfig.BoneVelocities.ToArray(),
                BoneRotations = initialConfig.BoneRotations.ToArray()
            };
            clonedRagdoll.initialPos = initialConfig.COMPosition;
            clonedRagdoll.clock = clock;
            clonedRagdoll.ControlSteps = 0;
            clonedRagdoll.TargetPosition = TargetPosition;
            if (this.Brain != null)
            {
                clonedRagdoll.SetBrain(this.Brain.Copy());
            }
            return clonedRagdoll;
        }

        public void EndOfGenerationTasks()
        {
            FinalizeCurrentPhase();
            Brain.ExecuteEndOfGenerationTasks();
        }

        public void reset()
        {
            // Restore every bone, not just the COM and rotations: otherwise a
            // deformed end-of-generation pose survives into the next evaluation.
            ResetAngularVelocities();
            for (int i = 0; i < Bones.Count; i++)
            {
                var rb = Bones[i].RB;
                rb.Position = initialConfig.BonePositions[i];
                rb.Orientation = initialConfig.BoneRotations[i];

                // Jitter throws if velocity is assigned to a static body, and a creature
                // authored in the builder can contain static bones.
                if (rb.MotionType != MotionType.Static)
                {
                    rb.Velocity = initialConfig.BoneVelocities[i];
                    rb.AngularVelocity = initialConfig.BoneAngularVelocities[i];
                }

                rb.Force = JVector.Zero;
                rb.Torque = JVector.Zero;
            }
            foreach (var joint in Joints)
            {
                joint.ResetWarmStart();
            }
            clock = 0f;
            EnergyUsed = 0f;
            ControlSteps = 0;
            COMPosition = ComputeCentreOfMass();
            BodyVelocity = ComputeVelocity();
            initialPos = initialConfig.COMPosition;
            TimeToReachTarget = 0f;
            reached = false;
            PhaseProgressMeters.Clear();
            PhaseStartDistances.Clear();
            PhaseTimesToReach.Clear();
            PhasesReachedCount = 0;
            hasActivePhase = false;
            phaseMinHorizontalDist = float.MaxValue;
            phaseReached = false;
            phaseClock = 0f;
            phaseStartClock = 0f;
            phaseStartDist = 0f;
        }

    }
}
