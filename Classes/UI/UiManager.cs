using _3D_Engine.Classes.Scenes;
using _3D_Engine.Classes.Signletons;
using ImGuiNET;
using System.Numerics;
using static System.Net.Mime.MediaTypeNames;

namespace _3D_Engine.Classes.UI
{
    public enum UiTypes
    {
        MainMenu,
        Simulation,
        Editor
    }

    public static class UiManager
    {
        public static ImGuiController guiController;
        public static Dictionary<string, string> UiTexts = new Dictionary<string, string>();
        public static UiTypes UiMode;

        public static List<float> FitnessGraphValues = new List<float>() { 0 };
        public static List<float> AvgFitnessGraphValues = new List<float>() { 0 };

        // Reused buffers for PlotLines. The previous code passed
        // `ref FitnessGraphValues.ToArray()[0]`, which either fails to compile or hands
        // ImGui a reference into a temporary that is immediately garbage, so the plot
        // never showed live data.
        private static float[] bestFitnessPlotBuffer = new float[1];
        private static float[] avgFitnessPlotBuffer = new float[1];

        public static void OnLoad()
        {
            //ImGui.SetNextWindowBgAlpha(1f);
        }

        public static void MainMenu()
        {

        }

        public static void SimulationUI()
        {

            ImGui.Begin("Simulation scene");
            //if (ImGui.Button("Save")) { /* ... */ }
            //ImGui.SliderFloat("Stiffness", ref stiffness, 0f, 200f);
            //ImGui.Text($"Generation: ");
            if (ImGui.BeginTabBar("bar1"))
            {
                if (ImGui.BeginTabItem("Statistics"))
                {
                    foreach (var text in UiTexts)
                    {
                        ImGui.Text($"{text.Key}: {text.Value}");
                    }
                    // Scale with the data, in both directions: fitness is unbounded and can
                    // be negative, so the old hardcoded 0..100 clipped the curve and made
                    // progress look flat.
                    bestFitnessPlotBuffer = CopyForPlot(FitnessGraphValues, bestFitnessPlotBuffer);
                    avgFitnessPlotBuffer = CopyForPlot(AvgFitnessGraphValues, avgFitnessPlotBuffer);

                    GetPlotRange(bestFitnessPlotBuffer, out float bestMin, out float bestMax);
                    GetPlotRange(avgFitnessPlotBuffer, out float avgMin, out float avgMax);

                    ImGui.PlotLines(
                        label: "Best Fitness",
                        values: ref bestFitnessPlotBuffer[0],
                        values_count: bestFitnessPlotBuffer.Length,
                        values_offset: 0,
                        overlay_text: "",
                        scale_min: bestMin,
                        scale_max: bestMax,
                        graph_size: new Vector2(0, 100) // 0 width fills available window space
                                           );
                    ImGui.PlotLines(
                        label: "Average Fitness",
                        values: ref avgFitnessPlotBuffer[0],
                        values_count: avgFitnessPlotBuffer.Length,
                        values_offset: 0,
                        overlay_text: "",
                        scale_min: avgMin,
                        scale_max: avgMax,
                        graph_size: new Vector2(0, 100) // 0 width fills available window space
                                           );

                    ImGui.EndTabItem();
                }
                if (ImGui.BeginTabItem("Sim Settings"))
                {
                    if (ImGui.Button("Change Sim Speed")) 
                    {
                        SimulationScene.simSpeedMultiplier = SimulationScene.simSpeedMultiplier == 1 ? 10 : (SimulationScene.simSpeedMultiplier == 10 ? 30 : 1);
                        SimulationProperties.game.VSync = SimulationScene.simSpeedMultiplier == 1
                            ? OpenTK.Windowing.Common.VSyncMode.On
                            : OpenTK.Windowing.Common.VSyncMode.Off;
                    }
                    ImGui.EndTabItem();

                    ImGui.Text($"Step per frame: {SimulationScene.simSpeedMultiplier}");
                }

                    ImGui.EndTabBar();

                //if ()
            }


            ImGui.End();



        }

        public static void EditorUI()
        {

        }

        public static void Update()
        {
            switch (UiMode)
            {
                case UiTypes.Simulation:
                    SimulationUI();
                    break;

                case UiTypes.Editor:
                    EditorUI(); 
                    break;

                default:
                    break;
            }


        }

        public static void ResetSimGraphs()
        {
            FitnessGraphValues = new List<float> { 0 };
            AvgFitnessGraphValues = new List<float> { 0 };
        }

        private static void GetPlotRange(float[] values, out float min, out float max)
        {
            min = 0f;
            max = 1f;
            bool first = true;

            for (int i = 0; i < values.Length; i++)
            {
                if (!float.IsFinite(values[i]))
                {
                    continue;
                }

                if (first)
                {
                    min = values[i];
                    max = values[i];
                    first = false;
                }
                else
                {
                    if (values[i] < min) min = values[i];
                    if (values[i] > max) max = values[i];
                }
            }

            // Keep a visible span so a flat curve does not collapse the axis.
            float span = max - min;
            float padding = MathF.Max(span * 0.1f, 1f);
            min -= padding;
            max += padding;
        }

        private static float[] CopyForPlot(List<float> values, float[] buffer)
        {
            if (values == null || values.Count == 0)
            {
                return buffer.Length > 0 ? buffer : new float[1];
            }

            if (buffer.Length != values.Count)
            {
                buffer = new float[values.Count];
            }

            values.CopyTo(buffer);
            return buffer;
        }
    }
}
