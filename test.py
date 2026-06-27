import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# 1. Load your saved predictions
data = np.load('output/predicted_poses_9_Directions 2.npy') 

# Extract the sequence of 27 frames
if len(data.shape) == 4:
    seq_data = data[0] # Shape becomes (27, 17, 3)
else:
    seq_data = data    # Shape is already (27, 17, 3)

num_frames = seq_data.shape[0]

# 2. Define the Human3.6M Skeleton connections
h36m_connections = [
    (0, 1), (1, 2), (2, 3),       # Right Leg
    (0, 4), (4, 5), (5, 6),       # Left Leg
    (0, 7), (7, 8), (8, 9), (9, 10), # Spine & Head
    (8, 11), (11, 12), (12, 13),  # Left Arm
    (8, 14), (14, 15), (15, 16)   # Right Arm
]

# 3. Setup the 3D Plot
fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111, projection='3d')

# Calculate GLOBAL limits so the camera doesn't bounce around during animation
x_max, x_min = seq_data[:, :, 0].max(), seq_data[:, :, 0].min()
y_max, y_min = seq_data[:, :, 1].max(), seq_data[:, :, 1].min()
z_max, z_min = seq_data[:, :, 2].max(), seq_data[:, :, 2].min()

max_range = np.array([x_max-x_min, y_max-y_min, z_max-z_min]).max() / 2.0
mid_x = (x_max+x_min) * 0.5
mid_y = (y_max+y_min) * 0.5
mid_z = (z_max+z_min) * 0.5

ax.set_xlim(mid_x - max_range, mid_x + max_range)
ax.set_ylim(mid_y - max_range, mid_y + max_range)
ax.set_zlim(mid_z - max_range, mid_z + max_range)
# ax.invert_zaxis() # Flip Z axis for standard visualization

ax.set_xlabel('X (mm)')
ax.set_ylabel('Y (mm)')
ax.set_zlabel('Z (mm)')

# Initialize empty scatter points and lines
scatter = ax.scatter([], [], [], c='red', s=50)
lines = [ax.plot([], [], [], color='blue', linewidth=2)[0] for _ in range(len(h36m_connections))]

# 4. The Animation Function
def update(frame):
    # Get coordinates for the current frame
    x = seq_data[frame, :, 0]
    y = seq_data[frame, :, 1]
    z = seq_data[frame, :, 2]
    
    # Update the joints
    scatter._offsets3d = (x, y, z)
    
    # Update the bones
    for i, bone in enumerate(h36m_connections):
        j1, j2 = bone
        # X and Y are set together, Z is set separately in 3D matplotlib
        lines[i].set_data([x[j1], x[j2]], [y[j1], y[j2]])
        lines[i].set_3d_properties([z[j1], z[j2]])
        
    ax.set_title(f'Predicted 3D Skeleton - Frame {frame+1} / {num_frames}')
    return [scatter] + lines

# 5. Play the animation
# interval=100 means 100 milliseconds between frames (~10 frames per second)
ani = animation.FuncAnimation(fig, update, frames=num_frames, interval=100, blit=False)

plt.show()