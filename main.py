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

    img = cv2.imread(img_path)
    img = vis_2D_pose(img, pose2D, data_type='mpii')
    img = cv2.resize(img, (256, 256))
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
