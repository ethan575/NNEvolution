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
                    ImGui.PlotLines(
                        label: "Best Fitness",
                        values: ref FitnessGraphValues.ToArray()[0],
                        values_count: FitnessGraphValues.Count,
                        values_offset: 0,
                        overlay_text: "",
                        scale_min: 0.0f,
                        scale_max: 100.0f,
                        graph_size: new Vector2(0, 100) // 0 width fills available window space
                                           );
                    ImGui.PlotLines(
                        label: "Average Fitness",
                        values: ref AvgFitnessGraphValues.ToArray()[0],
                        values_count: AvgFitnessGraphValues.Count,
                        values_offset: 0,
                        overlay_text: "",
                        scale_min: 0.0f,
                        scale_max: 100.0f,
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
    }
}
