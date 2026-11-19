using _3D_Engine.Classes.Ragdoll;
using Jitter2;
using Jitter2.Collision;
using Jitter2.Collision.Shapes;
using Jitter2.Dynamics;

namespace _3D_Engine.Classes.Scenes
{
    public sealed class RagdollSelfCollisionFilter : IBroadPhaseFilter
    {
        // Only bones that share a hinge are allowed to interpenetrate. The previous
        // version disabled *every* same-creature pair, so limbs passed straight through
        // the torso and evolution was free to exploit that.
        private readonly HashSet<long> adjacentPairs = new HashSet<long>();

        public RagdollSelfCollisionFilter(IEnumerable<Bone> bones, IEnumerable<Joint> joints)
        {
            foreach (Joint joint in joints)
            {
                if (joint.BoneA == null || joint.BoneB == null)
                {
                    continue;
                }

                adjacentPairs.Add(PairKey(joint.BoneA.RB, joint.BoneB.RB));
            }
        }

        private static long PairKey(RigidBody a, RigidBody b)
        {
            int low = Math.Min(a.GetHashCode(), b.GetHashCode());
            int high = Math.Max(a.GetHashCode(), b.GetHashCode());
            return ((long)low << 32) ^ (uint)high;
        }

        public bool Filter(IDynamicTreeProxy proxyA, IDynamicTreeProxy proxyB)
        {
            if (proxyA is RigidBodyShape shapeA && proxyB is RigidBodyShape shapeB)
            {
                if (adjacentPairs.Contains(PairKey(shapeA.RigidBody, shapeB.RigidBody)))
                {
                    return false;
                }
            }
            return true;
        }
    }

    public class HeadlessSimulationScene
    {

        public World world;
        public RagdollBody ragdollBody;

        public HeadlessSimulationScene(World world, RagdollBody ragdollBody)
        {
            this.world = world;
            this.ragdollBody = ragdollBody;

            world.AllowDeactivation = false;
            if (ragdollBody != null && ragdollBody.Bones.Count > 1)
            {
                world.BroadPhaseFilter = new RagdollSelfCollisionFilter(ragdollBody.Bones, ragdollBody.Joints);
            }
        }

    }
}
