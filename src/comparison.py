import numpy as np
import open3d as o3d
import math

mode = "comp"

n = 3

human_dir = "../data/human/"
ex_dir = "../paper/"
result_dir = "../paper/"

human_op = o3d.io.read_point_cloud(human_dir + "paper_1_human.ply")
human_op = human_op.voxel_down_sample(voxel_size=0.01)
human_np = np.asarray(human_op.points)

extraction_op = o3d.io.read_point_cloud(ex_dir + "7_extraction.ply")
extraction_np = np.asarray(extraction_op.points)

hrows, hcols = human_np.shape

human_np = np.round(human_np, decimals=n)
extraction_np = np.round(extraction_np, decimals=n)
np.savetxt(result_dir+"human_round.txt", human_np, fmt='%.4f')
np.savetxt(result_dir+"extraction_round.txt", extraction_np, fmt='%.4f')

dtype={'names':['f{}'.format(i) for i in range(hcols)], 'formats':hcols * [human_np.dtype]}

if mode == "comp":
    comp = np.intersect1d(human_np.view(dtype), extraction_np.view(dtype))
    comp = comp.view(human_np.dtype).reshape(-1, hcols)
    print(len(comp))
    print(len(human_np))
    print(len(comp)/len(human_np)*100)
    np.savetxt(result_dir+"comp.txt", comp)

elif mode == "back":
    comp = np.setdiff1d(extraction_np.view(dtype), human_np.view(dtype))
    comp = comp.view(human_np.dtype).reshape(-1, hcols)
    np.savetxt(result_dir+"back.txt", comp)
