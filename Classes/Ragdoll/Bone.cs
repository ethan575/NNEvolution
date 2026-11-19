using _3D_Engine.Classes.Objects;
using Jitter2;
using Jitter2.Collision.Shapes;
using Jitter2.Dynamics;
using Jitter2.LinearMath;
using OpenTK.Mathematics;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace _3D_Engine.Classes.Ragdoll
{
    public class Bone: RagdollObjects
    {
        public static int noOfBones = 0;

        public int BoneID { get; private set; }

        public string name { get; set; }
        //public RigidBody RB { get; set; }

        private const string BoneModelPath = "models/Raw/bone.stl";

        // The raw vertices never change, and the scaled cloud only depends on the bone's
        // visual scale, which is the same for every bone in a creature. Cloning 49 creatures
        // per generation used to re-read and re-parse bone.stl once per bone per clone.
        private static Vector3[] cachedRawBoneVertices;
        private static readonly Dictionary<(float, float, float), List<JVector>> cachedScaledVertices
            = new Dictionary<(float, float, float), List<JVector>>();

        private static List<JVector> GetScaledVertices(Vector3 scale)
        {
            (float, float, float) key = (scale.X, scale.Y, scale.Z);

            if (cachedScaledVertices.TryGetValue(key, out List<JVector> cached))
            {
                return cached;
            }

            if (cachedRawBoneVertices == null)
            {
                cachedRawBoneVertices = ModelImporter.LoadOnlyVertices(BoneModelPath);
            }

            List<JVector> verts = new List<JVector>(cachedRawBoneVertices.Length);
            foreach (Vector3 v in cachedRawBoneVertices)
            {
                verts.Add(new JVector(v.X * scale.X, v.Y * scale.Y, v.Z * scale.Z));
            }

            cachedScaledVertices[key] = verts;
            return verts;
        }

        public Bone(World world, WorldObject visual, Vector3 position, string name)   
        {
            this.BoneID = Bone.noOfBones;
            Bone.noOfBones++;

            this.Visual = visual;
            this.name = name;


            // Create a rigid body for the bone
            RB = world.CreateRigidBody();
            RB.Friction = 0.5f;
            //RB.AddShape(new BoxShape(shapeDims*0.5f));
            // Cached per visual scale; PointCloudShape copies the points it is given.
            List<JVector> verts = GetScaledVertices(visual.GetScale());
            RB.Position = new JVector(position.X, position.Y, position.Z);
            RB.Orientation = ToJQuaternion(visual.GetOrientation());
            RB.AddShape(new PointCloudShape(verts));
            // AddShape recalculates mass properties; set the desired mass afterwards.
            RB.SetMassInertia(1f);
            //RB.MotionType = MotionType.Dynamic;

        }

        public JVector getAngularVelocity()
        {
            return this.RB.AngularVelocity;
        }

        public override void UpdateVisual()
        {
            // Update the visual representation of the bone based on the rigid body's position and orientation
            Visual.setPosition(FromJVec(RB.Position));
            Visual.setRotation(FromJquaternion(RB.Orientation));
            Visual.Draw();
            //Console.WriteLine($"Bone {name} Rotation: ({rot.X}, {rot.Y}, {rot.Z}, {rot.W})");
            //Console.WriteLine($"Visual {name} Rotation: ({Visual.GetOrientation().X}, {Visual.GetOrientation().Y}, {Visual.GetOrientation().Z}, {Visual.GetOrientation().W})");


        }

        public void setBoneID(int id)
        {
            this.BoneID = id;
        }

        public override RagdollObjects Clone(Jitter2.World world)
        {
            WorldObject visualClone = Visual.Clone();

            Bone clone = new Bone(world, visualClone, FromJVec(RB.Position), name);
            clone.BoneID = this.BoneID;
            // Clone simulation state, never the potentially stale rendered transform.
            clone.RB.Orientation = RB.Orientation;
            clone.RB.SetMassInertia(RB.InverseInertia, 1f / RB.Mass, setAsInverse: true);
            clone.RB.MotionType = RB.MotionType;
            clone.RB.Friction = RB.Friction;
            clone.RB.Restitution = RB.Restitution;
            clone.RB.Damping = RB.Damping;
            clone.RB.AffectedByGravity = RB.AffectedByGravity;
            clone.RB.Velocity = RB.Velocity;
            clone.RB.AngularVelocity = RB.AngularVelocity;
            clone.RB.Force = RB.Force;
            clone.RB.Torque = RB.Torque;
            visualClone.setPosition(FromJVec(clone.RB.Position));
            visualClone.setRotation(FromJquaternion(clone.RB.Orientation));
            return clone;
        }

        static Vector3 FromJVec(JVector v)
        {
            return new Vector3(v.X, v.Y, v.Z);
        }
        static JVector ToJVec(Vector3 v)
        {
            return new JVector(v.X, v.Y, v.Z);
        }
        static Quaternion FromJquaternion(JQuaternion q)
        {
            // some random bug reqiures negative
            return new Quaternion(-q.X, q.Y, q.Z, -q.W);
        }
        static JQuaternion ToJQuaternion(Quaternion q)
        {
            return new JQuaternion(-q.X, q.Y, q.Z, -q.W);
        }

    }
}
