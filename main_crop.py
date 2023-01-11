import cv2
import numpy as np
from pose_data import pose2D_1, pose3D_1, pose3Dgt_1
from functions import *
import matplotlib.pyplot as plt


def final_visualization(pose_3D_pred, pose_3D_gt, arr, Radius, angles, save_img_folder):

    stepX, stepZ = Radius*2 / 256, Radius*2 / 256
    X1 = np.arange(-Radius, Radius, stepX)
    Z1 = np.arange(-Radius, Radius, stepZ)
    X1, Z1 = np.meshgrid(X1, Z1)

    fig = plt.figure(figsize=(8, 8))
    ax_3D_pred = fig.add_subplot(1, 1, 1, projection='3d')

    ax_3D_pred.plot_surface(X1, np.atleast_2d(
        Radius), Z1, rstride=1, cstride=1, facecolors=arr)

    ax_3D_pred.set_xlim3d([-Radius, Radius])
    ax_3D_pred.set_ylim3d([-Radius, Radius])
    ax_3D_pred.set_zlim3d([-Radius, Radius])
    ax_3D_pred.set_xticklabels([])
    ax_3D_pred.set_yticklabels([])
    ax_3D_pred.set_zticklabels([])
    ax_3D_pred.view_init(angles[0], angles[1])
    # ax_3D_pred.grid(False)

    ax_3D = fig.add_subplot(1, 1, 1, projection='3d')
    pred3D = pose_3D_pred - np.array([0, 0, 530])
    gt3D = pose_3D_gt - np.array([0, 0, 530])
    show3Dpose(pred3D, ax_3D, data_type='mpii',
               radius=Radius, lcolor='black', angles=angles)      # 20,-70
    # show3Dpose_with_annot2(gt3D, pred3D, ax_3D_pred, data_type='mpii', radius=Radius, lcolor='blac)
    ax_3D.grid(False)
    ax_3D.axis('off')
    print('drawing....')
    plt.draw()
    i = 0
    plt.savefig(save_img_folder + '/%05d.png' % (i), transparent=True)
    #plt.savefig(config.save_img_folder + '/%05d.png'% (i))
    plt.pause(0.01)
    plt.show()
    # fig.clear()


def main(img_path, pose2D, pose3D, pose3D_gt, save_img_folder):
    data_type = 'mpii'
    img = cv2.imread(img_path)
    img_with_p2d = vis_2D_pose(img, pose2D, data_type=data_type)

    # human crop
    x_set, y_set = pose2D.T

    # rectangles = [[] for _ in range(4)]
    rectangle = (max(int(abs(x_set.max()-x_set.min())),
                     int(abs(y_set.max()-y_set.min())))/2)*1.4
    # rectangles[v].append(rectangle)
    # rectangle = np.mean(rectangles[v])

    if data_type == 'h36m':
        root_joint = pose2D[0]
    else:
        root_joint = pose2D[6]

    left_range = int(root_joint[1]-rectangle)
    right_range = int(root_joint[1]+rectangle)
    bottom_range = int(root_joint[0]-rectangle)
    upper_range = int(root_joint[0]+rectangle)

    if left_range < 0:
        left_range = 0
    if right_range > img.shape[1]:
        right_range = img.shape[1]
    if bottom_range < 0:
        bottom_range = 0
    if upper_range > img.shape[0]:
        upper_range = img.shape[0]

    cropimg = img_with_p2d[left_range:
                           right_range,
                           bottom_range:
                           upper_range]

    img = cv2.resize(cropimg, (256, 256))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    arr = img / 255
    arr = cv2.flip(arr, 0)

    Radius = 200
    angles = (10, -70)

    final_visualization(pose3D, pose3D_gt, arr,
                        Radius, angles, save_img_folder)


if __name__ == '__main__':
    img_path = './mpi_inf_3dhp_test_set/TS2/imageSequence/000001.jpg'
    pose2D = pose2D_1
    pose3D = pose3D_1
    pose3D_gt = pose3Dgt_1
    save_img_folder = './output'
    main(img_path, pose2D, pose3D, pose3D_gt, save_img_folder)
