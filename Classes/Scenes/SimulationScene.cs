using _3D_Engine.Classes.Data;
using _3D_Engine.Classes.NeuralNet;
using _3D_Engine.Classes.Objects;
using _3D_Engine.Classes.physics;
using _3D_Engine.Classes.Ragdoll;
using _3D_Engine.Classes.UI;
using ImGuiNET;
using Jitter2;
using Jitter2.Collision.Shapes;
using Jitter2.Dynamics;
using Jitter2.LinearMath;
using Keys = OpenTK.Windowing.GraphicsLibraryFramework.Keys;
using Vector3 = OpenTK.Mathematics.Vector3;

namespace _3D_Engine.Classes.Scenes
{
    public class SimulationScene : Scene
    {
        private const float GenerationDuration = 20f;
        private const int TargetPhasesPerGeneration = 3;
        private const float TargetFloorY = -9f;

        // Target ring. Roughly the distance a walker can cover in one phase
        // (20 s / 3 phases = 6.67 s), so arrivals are actually achievable.
        private const float TargetRingMinRadius = 12f;
        private const float TargetRingMaxRadius = 18f;
        private const float MaxTargetRingExtent = 52f; // keep targets on the floor

        // The cube model is drawn at half scale, so its collider is built at half size too.
        private static readonly BoxShape TargetCubeShape = new BoxShape(0.5f);

        Game game;

        PhysicsObject floor;

        // jitter physics world
        List<HeadlessSimulationScene> AllSimWorlds = new List<HeadlessSimulationScene>();
        World ForegroundWorld = new World();
        // Separate world for the target marker so no creature can collide with it.
        World TargetWorld = new World();

        Shader shader;
        PhysicsServer physicsServer;

        RagdollBody ForegroundBody;

        PhysicsObject TargetCube;

        bool isFirstSimulationClick = true;
        bool brainLoadedFromFile = false;

        // simulation stuff
        bool isSimulationRunning = false;
        int GenNumber = 0;
        int populationSize = 50;
        RagdollBody LastLoadedCreature;
        private bool backgroundPopulationHidden = false;
        Generation currentGeneration = new Generation();
        string logFileName;
        public static int simSpeedMultiplier = 1;
        float dtAfterSimSpeed;
        private Random rng = new Random();

        private int StepsPerFrame;

        private int currentPhaseIndex = 0;
        private readonly JVector[] phaseTargets = new JVector[TargetPhasesPerGeneration];

        private TargetScheduleBank? activeBank;

        public SimulationScene(Game game) : base(game)
        {
            shader = game.shader;
            this.game = game;

            physicsServer = new PhysicsServer();

            logFileName = $"{DateTime.Now.ToString().Replace(":", "-").Replace("/", "-").Replace(" ", "_")}_training.log";

            createScene();
        }

        public override void Update(float dt)
        {
            if (isSimulationRunning && PhysicsServer.AllowedWorldStep)
            {
                int steps = simSpeedMultiplier == 1 ? 1 : simSpeedMultiplier;
                float stepDt = simSpeedMultiplier == 1
                    ? MathF.Min(dt, 1f / 30f)
                    : 1f / 60f;

                for (int step = 0; step < steps; step++)
                {
                    currentGeneration.ActiveTimeLeft -= stepDt;
                    UiManager.UiTexts["Generation Time left"] = Convert.ToString(
                        MathF.Truncate(MathF.Max(0f, currentGeneration.ActiveTimeLeft)));

                    if (currentGeneration.ActiveTimeLeft <= 0f)
                    {
                        PrepareEndGeneration();
                        break; // render & keep window responsive between generations
                    }

                    float elapsed = GenerationDuration - currentGeneration.ActiveTimeLeft;
                    float phaseDuration = GenerationDuration / TargetPhasesPerGeneration;
                    int desiredPhase = Math.Clamp((int)(elapsed / phaseDuration), 0, TargetPhasesPerGeneration - 1);

                    if (desiredPhase != currentPhaseIndex)
                    {
                        currentPhaseIndex = desiredPhase;
                        JVector nextTarget = phaseTargets[currentPhaseIndex];
                        SetTargetCubePosition(nextTarget);
                        foreach (var scene in AllSimWorlds)
                        {
                            scene.ragdollBody.StartTargetPhase(nextTarget);
                        }
                    }

                    JVector liveTargetPos = TargetCube.Rbody.Position;
                    Parallel.For(0, AllSimWorlds.Count, i =>
                    {
                        var scene = AllSimWorlds[i];
                        scene.ragdollBody.AdvanceSimulationClock(stepDt);
                        scene.ragdollBody.TargetPosition = liveTargetPos;
                        var inputM = scene.ragdollBody.GetBrainInputMatrix();
                        scene.ragdollBody.BrainNNFF(inputM);
                        scene.world.Step(stepDt);
                    });
                }

                // Sync visual transforms (floor, TargetCube) once at the end of the frame on the main thread
                foreach (var obj in AllPhysicsObjects)
                {
                    obj.Update(dt);
                }
            }
            else
            {
                physicsServer.Update(dt, AllSimWorlds, ForegroundWorld);
                AllSimWorlds.ForEach(s => s.ragdollBody.ResetAngularVelocities());
            }



            if (game.KeyboardState.IsKeyPressed(Keys.Escape))
            {
                SceneManager.LoadScene(new BuildScene(game));
                UiManager.ResetSimGraphs();
                return;
            }

            if (game.KeyboardState.IsKeyPressed(Keys.F))
            {
                simSpeedMultiplier = simSpeedMultiplier == 1 ? 10 : (simSpeedMultiplier == 10 ? 30 : 1);
                game.VSync = simSpeedMultiplier == 1
                    ? OpenTK.Windowing.Common.VSyncMode.On
                    : OpenTK.Windowing.Common.VSyncMode.Off;
            }

            if (game.KeyboardState.IsKeyDown(Keys.LeftControl) && game.keyboardState.IsKeyPressed(Keys.N))
            {
                if (!isFirstSimulationClick)
                {
                    MessageBox.Show("Simulation is already running. Return to BuildScene (Esc) and re-enter to load a different creature.", "Info", MessageBoxButtons.OK);
                }
                else
                {
                    using (var dialog = new OpenFileDialog())
                    {
                        dialog.Filter = "Creature File (*.creature)|*.creature";
                        dialog.Title = "Select a creature file";
                        if (dialog.ShowDialog() == DialogResult.OK)
                        {
                            string filePath = dialog.FileName;

                            if (ForegroundBody != null)
                            {
                                foreach (var b in ForegroundBody.Bones)
                                {
                                    b.Visual?.Destroy();
                                    ForegroundWorld.Remove(b.RB);
                                }
                                foreach (var j in ForegroundBody.Joints)
                                {
                                    j.destroy();
                                }
                                AllSimWorlds.Clear();
                            }

                            LastLoadedCreature = BuilderSaveReadWrite.BuilderLoadFile(filePath, ForegroundWorld, shader, this);
                            ForegroundBody = LastLoadedCreature;
                            ForegroundBody.initialPos = ForegroundBody.initialConfig.COMPosition;
                            brainLoadedFromFile = false;

                            AllSimWorlds.Add(new HeadlessSimulationScene(ForegroundWorld, ForegroundBody));
                        }
                    }
                }
            }

            if (game.KeyboardState.IsKeyDown(Keys.LeftControl) && game.keyboardState.IsKeyPressed(Keys.S))
            {
                if (currentGeneration.Population != null && currentGeneration.Population.Count > 0)
                {
                    NeuralNetwork NN = currentGeneration.Population.OrderByDescending(nn => nn.FitnessScore).First();

                    bool wasAllowed = PhysicsServer.AllowedWorldStep;
                    PhysicsServer.AllowedWorldStep = false;
                    using (var dialog = new SaveFileDialog())
                    {
                        dialog.Title = "Save NN as";
                        dialog.Filter = "NN File (*.nn)|*.nn";
                        dialog.DefaultExt = "nn";
                        dialog.AddExtension = true;
                        dialog.InitialDirectory = System.IO.Directory.GetCurrentDirectory();

                        if (dialog.ShowDialog() == DialogResult.OK)
                        {
                            string SavePath = dialog.FileName;
                            NNSaveLoad.SaveNeuralNetwork(NN, SavePath);
                            Console.WriteLine("Neural Network saved successfully.");
                        }
                    }
                    PhysicsServer.AllowedWorldStep = wasAllowed;
                }
            }

            // opening NN file
            if (game.KeyboardState.IsKeyDown(Keys.LeftControl) && game.keyboardState.IsKeyPressed(Keys.O))
            {
                bool wasAllowed = PhysicsServer.AllowedWorldStep;
                PhysicsServer.AllowedWorldStep = false;
                using (var dialog = new OpenFileDialog())
                {
                    dialog.Title = "Open NN file";
                    dialog.Filter = "NN File (*.nn)|*.nn";
                    dialog.DefaultExt = "nn";
                    dialog.AddExtension = true;
                    dialog.InitialDirectory = System.IO.Directory.GetCurrentDirectory();

                    if (dialog.ShowDialog() == DialogResult.OK)
                    {
                        string SavePath = dialog.FileName;

                        try
                        {
                            NeuralNetwork loaded = NNSaveLoad.LoadNeuralNetwork(SavePath);

                            if (ForegroundBody != null)
                            {
                                ForegroundBody.SetBrain(loaded);
                                brainLoadedFromFile = true;

                                if (AllSimWorlds.Count > 1)
                                {
                                    currentGeneration.Population = new List<NeuralNetwork>(AllSimWorlds.Count) { loaded };
                                    for (int i = 1; i < AllSimWorlds.Count; i++)
                                    {
                                        NeuralNetwork variant = NNTrainer.CreateMutatedVariant(loaded);
                                        AllSimWorlds[i].ragdollBody.SetBrain(variant);
                                        currentGeneration.Population.Add(variant);
                                    }
                                }
                                else
                                {
                                    currentGeneration.Population = new List<NeuralNetwork> { loaded };
                                }

                                Console.WriteLine("Neural Network loaded successfully.");
                            }
                            else
                            {
                                MessageBox.Show("There is no body to attach a brain to.", "Error", MessageBoxButtons.OK);
                            }
                        }
                        catch (Exception ex)
                        {
                            MessageBox.Show($"Failed to load neural network: {ex.Message}", "Error", MessageBoxButtons.OK);
                        }
                    }
                }
                PhysicsServer.AllowedWorldStep = wasAllowed;
            }

            if (game.KeyboardState.IsKeyPressed(Keys.Tab))
            {
                if (LastLoadedCreature == null)
                {
                    MessageBox.Show("No creature loaded. Please load a creature first.", "Error", MessageBoxButtons.OK);
                    return;
                }
                else if (ForegroundBody.Brain == null)
                {
                    int inputCount = ForegroundBody.RequiredInputCount;
                    int hiddenSize = Math.Max(16, inputCount);
                    NeuralNetwork newNN = new NeuralNetwork(new int[] { inputCount, hiddenSize, 16, ForegroundBody.Joints.Count });
                    ForegroundBody.SetBrain(newNN);
                }

                if (isFirstSimulationClick && LastLoadedCreature != null)
                {
                    isFirstSimulationClick = false;
                    PhysicsServer.AllowedWorldStep = true;
                    ForegroundBody.initialPos = ForegroundBody.initialConfig.COMPosition;

                    currentGeneration = new Generation(new List<NeuralNetwork>());
                    currentGeneration.ActiveTimeLeft = GenerationDuration;
                    currentGeneration.Population.Add(ForegroundBody.Brain);

                    for (int i = 0; i < populationSize - 1; i++) // '-1' because the first one is foreground
                    {
                        var newWorld = new World();

                        RigidBody floorRB = newWorld.CreateRigidBody();
                        floorRB.AddShape(new BoxShape(new JVector(120, 1f, 120)));
                        floorRB.MotionType = MotionType.Static;
                        floorRB.Position = new JVector(0, -10, 0);

                        RagdollBody newBody = LastLoadedCreature.Clone(newWorld);
                        NeuralNetwork bgBrain = brainLoadedFromFile
                            ? NNTrainer.CreateMutatedVariant(ForegroundBody.Brain)
                            : new NeuralNetwork(ForegroundBody.Brain.LayerDims);
                        newBody.SetBrain(bgBrain);
                        newBody.initialPos = newBody.initialConfig.COMPosition;

                        currentGeneration.Population.Add(bgBrain);
                        AllSimWorlds.Add(new HeadlessSimulationScene(newWorld, newBody));
                    }

                    AllSimWorlds.ForEach(scene => scene.ragdollBody.Joints.ForEach(joint => joint.Visual.visible = false));
                    AllSimWorlds.ForEach(scene => scene.ragdollBody.Bones.ForEach(bone => bone.Visual.visible = true));

                    initializeGenTargetSchedule();
                    isSimulationRunning = true;
                }
            }

            if (game.KeyboardState.IsKeyPressed(Keys.P))
            {
                for (int i = 0; i < AllSimWorlds.Count; i++)
                {
                    var scene = AllSimWorlds[i];
                    var com = scene.ragdollBody.COMPosition;
                    Console.WriteLine(
                        $"World {i}: bodies={scene.world.RigidBodies.Count} " +
                        $"COM=({com.X:F2}, {com.Y:F2}, {com.Z:F2}) " +
                        $"vel={scene.ragdollBody.BodyVelocity.Length():F2}");
                }
            }

            if (game.KeyboardState.IsKeyPressed(Keys.H))
            {
                for (int i = 1; i < AllSimWorlds.Count; i++)
                {
                    var scene = AllSimWorlds[i];
                    foreach (Bone bone in scene.ragdollBody.Bones)
                    {
                        bone.Visual.visible = backgroundPopulationHidden;
                    }
                }
                backgroundPopulationHidden = !backgroundPopulationHidden;
            }

            // Manual target nudging. The marker is static now (creatures must not be able to
            // push it), so these move it directly instead of injecting velocity - otherwise
            // the keys silently did nothing. During a generation the schedule owns the
            // target, so nudging is only allowed before the run starts.
            float u = 0.5f;

            if (isFirstSimulationClick && game.KeyboardState.IsKeyDown(Keys.W))
            {
                TargetCube.setPosition(TargetCube.GetPosition() + Vector3.UnitZ * u);
            }
            if (isFirstSimulationClick && game.KeyboardState.IsKeyDown(Keys.S))
            {
                TargetCube.setPosition(TargetCube.GetPosition() - Vector3.UnitZ * u);
            }
            if (isFirstSimulationClick && game.KeyboardState.IsKeyDown(Keys.A))
            {
                TargetCube.setPosition(TargetCube.GetPosition() - Vector3.UnitX * u);
            }
            if (isFirstSimulationClick && game.KeyboardState.IsKeyDown(Keys.D))
            {
                TargetCube.setPosition(TargetCube.GetPosition() + Vector3.UnitX * u);
            }
        }

        public override void OnFrameRender(float dt)
        {
            if (isFirstSimulationClick)
            {
                ForegroundBody?.Update(dt);
            }
            else
            {
                foreach (var scene in AllSimWorlds)
                {
                    scene.ragdollBody.Update(dt);
                }
            }

            foreach (var entity in SceneManager.RenderedEntities)
            {
                entity.Draw();
            }
        }

        public override void createScene()
        {
            base.createScene();

            float[] floorVertex = ModelImporter.LoadModel("models/Raw/floor.stl");
            float[] cubeVertex = ModelImporter.LoadModel("models/Raw/cube.stl");

            RigidBody floorRB = ForegroundWorld.CreateRigidBody();
            floorRB.AddShape(new BoxShape(new JVector(120, 1f, 120)));
            floorRB.MotionType = MotionType.Static;
            floor = new PhysicsObject("floor", shader, floorVertex, this, game, floorRB, new Vector3(1, 1, 1));
            floor.Scale(6, 0.5f, 6);
            floor.setPosition(new Vector3(0, -10, 0));

            // The target marker lives in its own world. It used to share ForegroundWorld
            // with the creatures, which meant the foreground creature (the one whose score
            // was "the elite") could physically shove or block the target, and every other
            // creature then steered at the moved cube.
            RigidBody RB1 = TargetWorld.CreateRigidBody();
            RB1.SetMassInertia(20);
            RB1.AddShape(TargetCubeShape);
            RB1.MotionType = MotionType.Static;
            RB1.AffectedByGravity = false;
            TargetCube = new PhysicsObject("convex1", shader, cubeVertex, this, game, RB1, new Vector3(0, 1, 0));
            // Note: the .stl is scaled 2x smaller than its collider, so the marker's visual
            // size stays what it always was on screen; only the collision extent changed.
            TargetCube.setMass(20f);
            TargetCube.setObeyGravity(false);
            TargetCube.Scale(0.25f);
            TargetCube.setPosition(5f, TargetFloorY, 35f);

            IsReady = true;
            PhysicsServer.AllowedWorldStep = false;
            UiManager.UiMode = UiTypes.Simulation;
        }

        public override void destroyScene()
        {
            IsReady = false;
            game.VSync = OpenTK.Windowing.Common.VSyncMode.On;

            foreach (var entity in new List<WorldObject>(SceneManager.RenderedEntities))
            {
                entity.Destroy();
            }

            foreach (var sim in AllSimWorlds)
            {
                if (sim.world != ForegroundWorld)
                {
                    sim.world.Clear();
                    sim.world.Dispose();
                }
            }
            AllSimWorlds.Clear();

            ForegroundWorld.Clear();
            ForegroundWorld.Dispose();

            TargetWorld.Clear();
            TargetWorld.Dispose();

            SceneManager.CurrentScene?.AllPhysicsObjects.Clear();
        }

        public void PrepareEndGeneration()
        {
            foreach (HeadlessSimulationScene scene in AllSimWorlds)
            {
                scene.ragdollBody.EndOfGenerationTasks();
            }

            for (int i = 0; i < AllSimWorlds.Count; i++)
            {
                var scene = AllSimWorlds[i];
                float fitnessScore = FitnessEval.EvaluateFitness(scene.ragdollBody);
                scene.ragdollBody.Brain.FitnessScore = fitnessScore;
            }

            currentGeneration.Population = currentGeneration.Population.OrderByDescending(nn => nn.FitnessScore).ToList();
            UiManager.UiTexts["BestFitness"] = $"{currentGeneration.Population[0].FitnessScore:F2}";

            UiManager.FitnessGraphValues.Add(currentGeneration.Population[0].FitnessScore);

            float sum = 0;
            int num = 0;
            currentGeneration.Population.ForEach(n =>
            {
                sum += n.FitnessScore;
                num++;
            });
            UiManager.AvgFitnessGraphValues.Add(sum / num);

            string log = $"Generation {GenNumber} - Best Fitness: {currentGeneration.Population[0].FitnessScore:F2} \n\n " +
                $"Weights: \n{string.Join("\n", currentGeneration.Population[0].NetworkWeights.Select(WeightM => WeightM.ToString()))}" +
                $"Bias: \n{string.Join("\n", currentGeneration.Population[0].ConstructBiasList().Select(BiasM => BiasM.ToString()))}";

            TextLogSave.AppendToTextLog(log, $"SAVES/logs/{logFileName}");

            currentGeneration = NNTrainer.EvolveGeneration(currentGeneration.Population);
            currentGeneration.ActiveTimeLeft = GenerationDuration;

            for (int i = 0; i < AllSimWorlds.Count; i++)
            {
                var scene = AllSimWorlds[i];
                scene.ragdollBody.SetBrain(currentGeneration.Population[i]);
            }

            StartNextGeneration();
        }

        public void StartNextGeneration()
        {
            GenNumber++;
            UiManager.UiTexts["Generation"] = $"{GenNumber}";

            AllSimWorlds.ForEach(scene =>
            {
                scene.ragdollBody.reset();
            });

            initializeGenTargetSchedule();
        }

        private void initializeGenTargetSchedule()
        {
            JVector centre = ForegroundBody != null
                ? ForegroundBody.initialConfig.COMPosition
                : JVector.Zero;

            // Scores are only comparable between generations if the task is comparable.
            // A bank of fixed schedules is pinned for a few generations and then rotated.
            TargetScheduleBank bank = NNTrainer.NextScheduleBank(activeBank, BuildTargetScheduleBank);
            activeBank = bank;
            JVector[] schedule = bank.Schedules[GenNumber % bank.Schedules.Length];

            JVector[] resolvedTargets = ResolveTargets(centre, schedule);

            for (int i = 0; i < TargetPhasesPerGeneration; i++)
            {
                phaseTargets[i] = resolvedTargets[i];
            }

            currentPhaseIndex = 0;
            SetTargetCubePosition(phaseTargets[0]);

            foreach (var scene in AllSimWorlds)
            {
                scene.ragdollBody.StartTargetPhase(phaseTargets[0]);
            }
        }

        /// <summary>
        /// Builds a bank of schedules as offsets from the spawn point. Each schedule turns
        /// the creature through <see cref="TargetPhasesPerGeneration"/> directions that are
        /// evenly spread around the circle and then shuffled, so the network cannot read a
        /// fixed turn order off its clock.
        /// </summary>
        private JVector[][] BuildTargetScheduleBank()
        {
            JVector[][] schedules = new JVector[TargetScheduleBank.SchedulesPerBank][];

            for (int s = 0; s < schedules.Length; s++)
            {
                schedules[s] = new JVector[TargetPhasesPerGeneration];

                float baseAngle = rng.NextSingle() * MathF.PI * 2f;
                float[] angles = new float[TargetPhasesPerGeneration];
                for (int i = 0; i < TargetPhasesPerGeneration; i++)
                {
                    angles[i] = baseAngle + i * (MathF.PI * 2f / TargetPhasesPerGeneration);
                }

                for (int i = angles.Length - 1; i > 0; i--)
                {
                    int j = rng.Next(i + 1);
                    (angles[i], angles[j]) = (angles[j], angles[i]);
                }

                for (int i = 0; i < TargetPhasesPerGeneration; i++)
                {
                    float radius = TargetRingMinRadius
                        + rng.NextSingle() * (TargetRingMaxRadius - TargetRingMinRadius);

                    schedules[s][i] = new JVector(MathF.Cos(angles[i]), 0f, MathF.Sin(angles[i])) * radius;
                }
            }

            return schedules;
        }

        private static JVector[] ResolveTargets(JVector centre, JVector[] offsets)
        {
            // Clamp the *offset length*, not X and Z independently. Independent clamping
            // pulled targets toward whichever axis was clipped, so an off-centre spawn gave
            // some world directions a shorter target than others - a stochastic reward for
            // facing one way, which is what made the population pick a favourite corner.
            float maxOffset = MaxTargetRingExtent - MathF.Max(MathF.Abs(centre.X), MathF.Abs(centre.Z));
            if (maxOffset < 1f) maxOffset = 1f;

            JVector[] resolved = new JVector[offsets.Length];
            for (int i = 0; i < offsets.Length; i++)
            {
                JVector offset = offsets[i];
                if (offset.Length() > maxOffset)
                {
                    offset *= maxOffset / offset.Length();
                }

                resolved[i] = new JVector(centre.X + offset.X, TargetFloorY, centre.Z + offset.Z);
            }

            return resolved;
        }

        private void SetTargetCubePosition(JVector pos)
        {
            // Teleport-only marker: static bodies accept direct position writes and update
            // the broadphase, so the cube can never be pushed off its scheduled point.
            TargetCube.Rbody.MotionType = MotionType.Static;
            TargetCube.setPosition(pos.X, pos.Y, pos.Z);
            TargetCube.setVelocity(Vector3.Zero);
            TargetCube.Rbody.AngularVelocity = JVector.Zero;
        }
    }
}
