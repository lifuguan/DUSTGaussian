import numpy as np
import os
import xml.etree.ElementTree as ET

def intrinsics_from_xml(xml_file):
    root = ET.parse(xml_file).getroot()
    calibration = root.find('chunk/sensors/sensor/calibration')
    resolution = calibration.find('resolution')
    width = float(resolution.get('width'))
    height = float(resolution.get('height'))
    f = float(calibration.find('f').text)
    cx = width/2
    cy = height/2

    K = np.array([
        [f, 0, cx],
        [0, f, cy],
        [0, 0,  1]
    ], dtype=np.float32)

    return K, (width, height)

def extrinsics_from_xml(xml_file, verbose=False):
    root = ET.parse(xml_file).getroot()
    transforms = {}
    for e in root.findall('chunk/cameras')[0].findall('camera'):

        label = e.get('label')
        try:
            transforms[label] = e.find('transform').text
        except:
            if verbose:
                print('failed to align camera', label)

    view_matrices = []
    # labels_sort = sorted(list(transforms), key=lambda x: int(x))
    labels_sort = list(transforms)
    for label in labels_sort:
        extrinsic = np.array([float(x)
                             for x in transforms[label].split()]).reshape(4, 4)
        extrinsic[:, 1:3] *= -1
        view_matrices.append(extrinsic)

    return view_matrices, labels_sort

def read_xml(data_dir):
    print("Parsing Metashape results")
    intrinsics = intrinsics_from_xml(os.path.join(data_dir, 'pose.xml'))
    extrinsics, labels_sort = extrinsics_from_xml(os.path.join(data_dir, 'pose.xml'))
    for label, extrinsic in zip(labels_sort, extrinsics):
        np.savetxt(os.path.join(data_dir, 'extrinsics', f'{label}.txt'), extrinsic)

    intrinsic = np.identity(4)
    intrinsic[:3, :3] = intrinsics[0]
    np.savetxt(os.path.join(data_dir, 'intrinsic.txt'), intrinsic)
    print("Finished.")

    # poses = extrinsic
    # poses = np.concatenate(
    #     (-poses[:, :, 1:2], poses[:, :, 0:1], poses[:, :, 2:3], poses[:, :, 3:]), axis=-1)



if __name__ == '__main__':
    read_xml('data/baidu_map/003')