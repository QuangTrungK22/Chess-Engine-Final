import open3d as o3d
import numpy as np

# Draw lane lines as 3D polylines
points = np.array([[0,0,0], [1,1,0], [2,1.5,0]])  # lane points
lines = [[i, i+1] for i in range(len(points)-1)]
colors = [[1, 0, 0] for _ in lines]  # red lines

line_set = o3d.geometry.LineSet(
    points=o3d.utility.Vector3dVector(points),
    lines=o3d.utility.Vector2iVector(lines)
)
line_set.colors = o3d.utility.Vector3dVector(colors)

o3d.visualization.draw_geometries([line_set])
