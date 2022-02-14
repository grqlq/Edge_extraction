import numpy as np
import numpy.linalg as LA
import open3d as o3d
import matplotlib.pyplot as plt
import sys
import os
import shutil
import time

start = time.time()

ply = o3d.io.read_point_cloud("../data/input/paper_1_rot.ply")

output_dir = "../paper/"

# define variable
k_n = 50
thresh = 0.04
voxel = 0.01
bins = 60
number = 2000
minwidth = 0.4
maxwidth = 2.0
minheight = 1.3
maxheight = 1.9
minlength = 0.1
maxlength = 0.4

if not os.path.exists(output_dir):
    os.makedirs(output_dir)
else:
    shutil.rmtree(output_dir)
    os.mkdir(output_dir)

#downsample
ply = ply.voxel_down_sample(voxel_size=voxel)
o3d.io.write_point_cloud(output_dir+"down.ply", ply)

# floor filter
ply_np = np.asarray(ply.points)
hist = np.histogram(ply_np[:,2], bins=bins)
floor = np.array((np.where(hist[1][np.argmax(hist[0])+1]>ply_np[:,2]))).flatten()
fil_z = np.delete(ply_np, floor, axis=0)

np.savetxt(output_dir+"fil.txt", fil_z[:,:3])

ply = o3d.geometry.PointCloud()
ply.points = o3d.utility.Vector3dVector(fil_z)

# extraction edge
ply.estimate_covariances(search_param = o3d.geometry.KDTreeSearchParamKNN(knn = k_n))
ply_cov_np = np.asarray(ply.covariances)
w, v = LA.eig(ply_cov_np)
e = np.sort(w)

sum_eg = np.sum(e, axis=1)

sigma_value = np.divide(e[:,0], sum_eg).reshape(len(ply.points),1)
ply_np = np.concatenate([fil_z, sigma_value], 1)

flat = np.array(np.where(ply_np[:,3]<thresh)).flatten()
edge = np.delete(ply_np, flat, axis=0)
np.savetxt(output_dir+"edge.txt", edge[:,:3])

# touch points
hist_edge = np.histogram(edge[:,1])
hist_edge_max = hist_edge[1][np.argmax(hist_edge[0])+1]
hist_edge_min = hist_edge[1][np.argmax(hist_edge[0])]
touch = np.array((np.where((hist_edge_max>edge[:,1])&(hist_edge_min<edge[:,1])))).flatten()
touch_edge = np.array(edge[touch])

np.savetxt(output_dir+"touch_edge.txt", touch_edge[:,:3])

#extraction without wall
wo_np = np.array((np.where(hist_edge_min>fil_z[:,1]))).flatten()
wo_np = np.array(fil_z[wo_np])
np.savetxt(output_dir+"wo_touch_edge.txt", wo_np[:,:3])

#DBSCAN clustering
wo = o3d.geometry.PointCloud()
wo.points = o3d.utility.Vector3dVector(wo_np[:,:3])

with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Debug) as cm:
    labels = np.array(wo.cluster_dbscan(eps=0.02, min_points=10, print_progress=True))

ply_np = np.concatenate([wo_np, labels.reshape(len(wo_np), 1)], 1)
max_label = labels.max()
print(f"point cloud has {max_label + 1} clusters")

colors = plt.get_cmap("tab20")(labels / (max_label if max_label > 0 else 1))
colors[labels < 0] = 0
wo.colors = o3d.utility.Vector3dVector(colors[:, :3])

o3d.io.write_point_cloud(output_dir+"DBSCAN.ply", wo)

#human extraction
candidate = np.empty((0,4))
count=0
for i in range(max_label):
    clu = np.array(np.where(ply_np[:,3]==i)).flatten()
    if len(clu) >= number:
        candidate = np.vstack([candidate, ply_np[clu]])
        count += 1
print("count:",count)
if count == 1:
    np.savetxt(output_dir+"extraction.txt", candidate[:,:3])
    human_op = o3d.geometry.PointCloud()
    human_op.points = o3d.utility.Vector3dVector(candidate[:,:3])
    o3d.io.write_point_cloud(output_dir+"extraction.ply", human_op)
    end = time.time()
    process = end - start
    print(process)
    sys.exit()

flag = np.zeros(0)
for i in range(int(candidate[np.argmax(candidate[:,3]), 3])+1):
    cluster = np.array(np.where(candidate[:,3]==i)).flatten()
    if len(cluster)==0:
        continue
    candidate_clu = candidate[cluster]
    candidate_on = o3d.geometry.PointCloud()
    candidate_on.points = o3d.utility.Vector3dVector(candidate_clu[:,:3])
    candidate_box = o3d.geometry.AxisAlignedBoundingBox.create_from_points(candidate_on.points)
    box = candidate_box.get_extent()
    judge = np.where((box[0]>=minwidth) & (box[0]<=maxwidth) & (box[1]>=minlength) & (box[1]<=maxlength) & (box[2]>=minheight) & (box[2]<=maxheight), True, False)
    if judge == True:
        flag = np.hstack([flag, i])

for i in range(len(flag)):
    human = candidate[np.where(candidate[:,3]==flag[i])]

np.savetxt(output_dir+"extraction.txt", human[:,:3])
human_op = o3d.geometry.PointCloud()
human_op.points = o3d.utility.Vector3dVector(human[:,:3])
o3d.io.write_point_cloud(output_dir+"extraction.ply", human_op)

end = time.time()
process = end - start
print(process)
