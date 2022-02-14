import numpy as np
import open3d as o3d
import math

mode1 = "comp"
mode2 = "txt"
n = 3

human_dir = "../data/human/"
ex_dir = "../paper/"
result_dir = "../paper/"

human_op = o3d.io.read_point_cloud(human_dir + "paper_4_human.ply")
#human_op = human_op.voxel_down_sample(voxel_size=0.01)
human = np.asarray(human_op.points)
#human = np.loadtxt(human_dir+".txt")
#extraction = np.loadtxt(ex_dir+"human.txt")
extraction_op = o3d.io.read_point_cloud(ex_dir + "extraction.ply")
extraction = np.asarray(extraction_op.points)

hrows, hcols = human.shape
"""
for i in range(hrows):
    for j in range(hcols):
        before = human[i, j]
        after = math.floor(before*10**n)/(10**n)
        human[i, j] = after

erows, ecols = extraction.shape
for i in range(erows):
    for j in range(ecols):
        before = extraction[i, j]
        after = math.floor(before*10**n)/(10**n)
        extraction[i, j] = after
"""
human = np.round(human, decimals=n)
extraction = np.round(extraction, decimals=n)
np.savetxt(result_dir+"human_round.txt", human, fmt='%.4f')
np.savetxt(result_dir+"extraction_round.txt", extraction, fmt='%.4f')

"""
comp = np.empty((0,3))
for i in range(len(human)):
    for j in range(len(extraction)):
        c = (human[i]==extraction[j]).all()
        if c==True:
            comp = np.vstack([comp, extraction[j].reshape(1,3)])
            continue
print(len(comp))
"""
dtype={'names':['f{}'.format(i) for i in range(hcols)], 'formats':hcols * [human.dtype]}

if mode1 == "comp":
    comp = np.intersect1d(human.view(dtype), extraction.view(dtype))
    comp = comp.view(human.dtype).reshape(-1, hcols)
    print(len(comp))
    print(len(human))
    print(len(comp)/len(human)*100)
    np.savetxt(result_dir+"comp.txt", comp)

elif mode1 == "back":
    comp = np.setdiff1d(extraction.view(dtype), human.view(dtype))
    comp = comp.view(human.dtype).reshape(-1, hcols)
    np.savetxt(result_dir+"back.txt", comp)