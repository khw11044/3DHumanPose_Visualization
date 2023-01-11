import cv2
import numpy as np
H36M_JOINTMAP = [
    [0, 1], [1, 2], [2, 3],       # 오른쪽 하체         # 0번째가 root joint
    [0, 4], [4, 5], [5, 6],        # 왼쪽 하체
    [7, 13], [13, 14], [14, 15],     # 오른쪽 상체
    [7, 10], [10, 11], [11, 12],     # 왼쪽 하체
    [0, 7], [7, 8], [8, 9],          # 중앙 상체
]

MPII_JOINTMAP = [
    [0, 1], [1, 2], [2, 6],     # 오른쪽 하체           # 6번째가 root joint
    [6, 3], [3, 4], [4, 5],      # 왼쪽 하체
    [10, 11], [11, 12], [12, 7],
    [7, 13], [13, 14], [14, 15],
    [6, 7], [7, 8], [8, 9],      # 중앙 상체
]

color_box = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]


def vis_2D_pose(frame, denorm_2d_pose, data_type):
    x_set, y_set = denorm_2d_pose.T

    if data_type == 'h36m':
        JOINTMAP = H36M_JOINTMAP
        root_joint_numer = 0
    else:
        JOINTMAP = MPII_JOINTMAP
        root_joint_numer = 6

    for j in range(len(JOINTMAP)):

        child = tuple([int(x_set[JOINTMAP[j][0]]), int(y_set[JOINTMAP[j][0]])])
        parent = tuple([int(x_set[JOINTMAP[j][1]]),
                       int(y_set[JOINTMAP[j][1]])])
        if j in [0, 1, 2, 6, 7, 8]:          # 오른쪽
            color = color_box[0]
        elif j in [3, 4, 5, 9, 10, 11]:       # 왼쪽
            color = color_box[2]
        else:
            color = color_box[1]        # 중앙
        # color = (0, 64 + 192 * (len(JOINTMAP) - j) / (len(JOINTMAP) - 1), 64 + 192 * j / (len(JOINTMAP) - 1))

        size = 20
        lsize = 20
        cv2.line(frame, child, parent, color, lsize)
        # cv2.circle(frame, parent, size, (0, 0, 0), -1)
    # for jnum in range(denorm_2d_pose.shape[-1]):
    #     #jnum = 10
    #     cv2.circle(frame, (int(x_set[jnum]), int(
    #         y_set[jnum])), size, (0, 255, 255), -1)
    return frame


def show3Dpose(annot, ax, radius=40, data_type='h36m', lcolor='red', rcolor='#0000ff', angles=(10, -60)):
    l_size = 6
    if data_type == 'h36m':
        JOINTMAP = H36M_JOINTMAP
        root_joint_numer = 0
    else:
        JOINTMAP = MPII_JOINTMAP
        root_joint_numer = 6

    for ind, (i, j) in enumerate(JOINTMAP):
        if ind in [0, 1, 2, 6, 7, 8]:          # 오른쪽
            color = 'b'
        elif ind in [3, 4, 5, 9, 10, 11]:       # 왼쪽
            color = 'r'
        else:
            color = 'g'        # 중앙

        x, z, y = [np.array([annot[i, c], annot[j, c]]) for c in range(3)]
        ax.plot(x, y, -z, lw=l_size, c=color)

    RADIUS = radius  # space around the subject

    xroot, yroot, zroot = annot[root_joint_numer,
                                0], annot[root_joint_numer, 1], annot[root_joint_numer, 2]

    ax.set_xlim3d([-RADIUS, RADIUS])
    ax.set_ylim3d([-RADIUS, RADIUS])
    ax.set_zlim3d([-RADIUS, RADIUS])

    # ax.set_xlim3d([-RADIUS + xroot, RADIUS + xroot])
    # ax.set_ylim3d([-RADIUS + yroot, RADIUS + yroot])
    # ax.set_zlim3d([-RADIUS + zroot, RADIUS + zroot])
    ax.view_init(angles[0], angles[-1])
    # ax.set_xlabel("x")
    # ax.set_ylabel("y")
    # ax.set_zlabel("z")

    # ax.set_xticklabels([])
    # ax.set_yticklabels([])
    # ax.set_zticklabels([])
