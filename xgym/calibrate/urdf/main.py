# mpl webagg
import matplotlib as mpl
from tqdm import tqdm

mpl.use("webagg")
import matplotlib.pyplot as plt
import numpy as np
import pytorch_kinematics as pk
from rich.pretty import pprint

from pathlib import Path

# name of this file
FNAME = Path(__file__)
DNAME = Path(__file__).parent

urdf = DNAME / "xarm7_standalone.urdf"
# there are multiple natural end effector links so it's not a serial chain
chain = pk.build_chain_from_urdf(open(urdf, mode="rb").read())
# visualize the frames (the string is also returned)
chain.print_tree()

# extract a specific serial chain such as for inverse kinematics
serial_chain = pk.SerialChain(chain, "link_tcp", "link_base")
serial_chain.print_tree()

# prints out list of joint names
print(chain.get_joint_parameter_names())

th = np.array([0.0, -np.pi / 4.0, 0.0, np.pi / 2.0, 0.0, np.pi / 4.0, 0.0])

fig, ax = plt.subplots(1, 1, figsize=(10, 10), subplot_kw={"projection": "3d"})

allpositions = {}
positions = []
scale = 0.5
scale = 2
for j in tqdm(joints := range(7)):
    for i in np.linspace(-np.pi / scale, np.pi / scale, 50):
        t = th.copy()
        t[j] += i

        ret = serial_chain.forward_kinematics(t, end_only=False)
        tg = ret["link_tcp"]
        m = tg.get_matrix()
        pos = m[:, :3, 3]
        rot = pk.matrix_to_quaternion(m[:, :3, :3])
        # print(f"joint {j} angle {i:.2f} pos {pos} rot {rot}")

        ap = {
            k: ret[k].get_matrix()[:, :3, 3].flatten().numpy()
            for k in ret.keys()
            if k.startswith("link")
        }
        for k, v in ap.items():
            if k not in allpositions:
                allpositions[k] = []
            allpositions[k].append(v)

        # convert to x,y,z
        positions.append(pos.flatten().numpy())


allpositions = {k: np.array(v) for k, v in allpositions.items()}
for k, v in allpositions.items():
    x, y, z = v[:, 0], v[:, 1], v[:, 2]
    ax.scatter(x, y, z, marker="o", label=k)

# scatter link_base with large circle
base = allpositions["link_base"]
x, y, z = base[:, 0], base[:, 1], base[:, 2]
ax.scatter(x, y, z, c="b", marker="x", label="Base")

# plot all joints together
keys = [
    "link_base",
    "link1",
    "link2",
    "link3",
    "link4",
    "link5",
    "link6",
    "link7",
    "link_eef",
    "link_tcp",
]
for i in range(0, len(base), 10):
    x = [allpositions[k][i, 0] for k in keys]
    y = [allpositions[k][i, 1] for k in keys]
    z = [allpositions[k][i, 2] for k in keys]
    ax.plot(x, y, z, alpha=0.8, color="gray")

positions = np.array(positions)
x, y, z = positions[:, 0], positions[:, 1], positions[:, 2]
# ax.scatter(x, y, z, c="r", marker="o")

ax.legend()
plt.show()
quit()


# specify joint values (can do so in many forms)
# do forward kinematics and get transform objects; end_only=False gives a dictionary of transforms for all links
ret = serial_chain.forward_kinematics(th, end_only=False)
# look up the transform for a specific link

# numpy no scientific notation
np.set_printoptions(suppress=True)

pprint(ret)
tg = ret["link_tcp"]
# get transform matrix (1,4,4), then convert to separate position and unit quaternion
m = tg.get_matrix()
pos = m[:, :3, 3]
rot = pk.matrix_to_quaternion(m[:, :3, :3])

pprint(m.numpy())
