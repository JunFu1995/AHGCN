import os
import six
import torch
import random
import numbers
import numpy as np
import cv2
import math
import scipy.io as scio
from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

import torch.utils.data as data
from torchvision import transforms
#import datasets.transforms as mytransforms
import torchvision.transforms.functional as tF
# import utils
import sys
if sys.version_info[0] == 2:
    import cPickle as pickle
else:
    import pickle


def get_dataset(is_training):
    if is_training:
        img_name_list = ['094', '082', '437', '532', '223', '447', '162', '354', '387', '072', '382', '194', '170', '457', '215', '226', '076', '432', 
        '511', '262', '156', '451', '175', '534', '421', '079', '081', '244', '087', '228', '461', '011', '467', '352', '294', 
        '180', '433', '122', '130', '128', '393', '476', '468', '013', '145', '381', '149', '513', '526', '086', '187', '099', 
        '410', '378', '188', '191', '243', '458', '415', '015', '218', '159', '427', '281', '267', '077', '538', '472', '163', 
        '189', '164', '222', '423', '271', '004', '160', '537', '351', '359', '030', '364', '084', '147', '109', '372', '025', 
        '474', '521', '288', '115', '266', '542', '400', '287', '237', '165', '369', '445', '206', '092', '127', '525', '349', '396', 
        '383', '069', '095', '090', '460', '028', '231', '143', '227', '416', '519', '240', '473', '073', '523', '414', '204', '217', 
        '151', '269', '012', '209', '544', '213', '357', '373', '275', '386', '088', '518', '071', '539', '514', '346', '010', '255',
        '404', '291', '374', '361', '441', '103', '297', '527', '522', '202', '169', '446', '135', '264', '021', '401', '392', '225', 
        '201', '409', '362', '305', '199', '230', '022', '214', '235', '306', '459', '083', '283', '298', '290', '137', '257', '258', '397', 
        '406', '133', '186', '443', '528', '475', '417', '444', '205', '540', '234', '448', '250', '391', '516', '370', '530', '376', '185', '144', '179', '375', 
        '023', '428', '241', '123', '110', '154', '389', '174', '220', '449', '367', '177', '246', '106', '129', '112', '032', '173', '247', '150', '384', '261', 
        '420', '436', '096', '211', '517', '398', '207', '208', '345', '286', '300', '355', '399', '224', '430', '365', '146', '347', '018', '105', '429', '210', 
        '408', '104', '031', '254', '120', '341', '003', '251', '253', '263', '385', '424', '422', '111', '270', '533', '285', '020', '439', '152', '153', '089', 
        '008', '216', '119', '101', '155', '284', '543', '117', '273', '356', '463', '157', '377', '184', '303', '535', '301', '019', '371', '464', '212', '233', 
        '512', '531', '070', '245', '196', '302', '342', '005', '541', '470', '248', '168', '299', '363', '221', '192', '181', '394', '116', '172', '453', '182', 
        '440', '085', '140', '026', '232', '125', '167', '435', '161', '289', '412', '353', '176', '282', '080', '450', '136', '407', '034', '403', '413', '121', 
        '454', '166', '158', '520', '131', '134', '280', '098', '148', '344', '238', '113', '405', '171', '190', '419', '006', '343', '007', '276', '002', '411', 
        '100', '418', '203', '360', '178', '016', '027', '138', '295', '465', '529', '265', '108', '249', '256', '074', '388', '229', '402', '114', '469', '033', 
        '242', '126', '442', '141', '358', '466', '296', '379', '536', '431', '075', '097', '471', '118', '193', '434', '390', '348', '368', '024', '198', '456', 
        '272', '350', '252', '017', '139', '274', '195', '380', '107', '366', '142', '093', '395', '091', '132', '009', '001', '279', '278', '029', '014', '455', 
        '124', '260', '438', '102', '078', '293', '239', '259', '183', '462', '304', '236', '292', '277', '197', '426', '524', '268', '200', '452', '515', '425', '219']
    else:
        img_name_list = ['497', '500', '495', '064', '332', '486', '483', '326', '050', '055', '488', '505', '506', '337', '324', '317', '312', '037', '330', '061', '498', '058', '335', '327', '485', '491', '484', '493', '482', '325', '499', '314', '063', '066', '339', '478', '502', '490', '340', '321', '510', '311', '309', '068', '336', '503', '319', '308', '492', '053', '489', '487', '322', '047', '504', '480', '318', '501', '065', '494', '479', '331', '043', '507', '508', '315', '323', '036', '035', '333', '051', '328', '040', '334', '062', '038', '044', '048', '056', '039', '054', '477', '052', '320', '496', '042', '057', '041', '316', '313', '046', '067', '338', '060', '310', '307', '045', '329', '481', '049', '059', '509']

    img_path, transform, is_training, label_path = '/home/fujun/datasets/iqa/CVIQ_Database/fov/',  None, False, '/home/fujun/datasets/iqa/CVIQ_Database/'
    ds = ImgDataset(img_path, img_name_list, transform, is_training, label_path)
    
    return  ds

def get_setspath(is_training):
    sets_root = './database/'
    if is_training:
        sets = [
            'cviqd_resize_imgtrain'
        ]
    else:
        sets = [
            'cviqd_resize_imgtest'
        ]
    return [os.path.join(sets_root, set) for set in sets]

def get_labelspath(is_training):
    sets_root = './database/'
    if is_training:
        sets = [
            'cviqd_fovall_label'
        ]
    else:
        sets = [
            'cviqd_fovall_label'
        ]
    return [os.path.join(sets_root, set) for set in sets]


def get_transform():
    return transforms.Compose([
        transforms.ToTensor()
    ])


class ImgDataset(data.Dataset):
    def __init__(self, root, img_name, transform, is_training, label_path, shuffle=False):
        self.root = root
        self.img_name = img_name

        # load label 
        data = scio.loadmat(os.path.join(label_path, 'CVIQ.mat'))['CVIQ']
        label = {}
        for i in range(data.shape[0]):
            imgname, score = data[i][0][0], data[i][1][0][0]
            imgname = imgname.split('.')[0]
            #print(imgname, score)
            label[imgname] = score 
        self.label = label 

        self.transform = transform

        self.fovNum = 20 

        # self.nSamples = len(self.img_path)
        # self.indices = range(self.nSamples)
        # if shuffle:
        #     random.shuffle(self.indices)
        # self.transform = transform
        # self.is_training = is_training
        # self.label_path = label_path

    def angulardist(self, point1, point2):
        lon1, lat1 = point1
        lon2, lat2 = point2
        angle = math.acos(math.cos(lat1)*math.cos(lat2)*math.cos(lon2-lon1)+math.sin(lat1)*math.sin(lat2))
        return angle / math.pi * 180 
    def calculate_adj_mtx(self, coord):
        #adj_mtx = np.zeros((self.fovNum,self.fovNum))
        adj_mtx = np.eye(self.fovNum)
        for i in range(self.fovNum-1):
            for j in range(i+1, self.fovNum):
                if abs(self.angulardist(coord[i], coord[j])) <= 45: 
                    adj_mtx[i][j] = 1
                    adj_mtx[j][i] = 1
        return adj_mtx 

    def __getitem__(self, index):

        imgname = self.img_name[index]
        img_group = []
        for i in range(self.fovNum):
            imgpath = os.path.join(self.root, imgname, '%s_fov%d.png'%(imgname, i+1))
            img = cv2.imread(imgpath)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = np.transpose(img, (2, 0, 1))
            img_group.append(img)
        img_group = np.array(img_group)

        # calculate graph structure 
        coordpath = os.path.join(self.root, imgname, '%s_coord.mat'%imgname)
        coord = scio.loadmat(coordpath)['spoint_radian']
        #print(coord)
        adj_mtx = self.calculate_adj_mtx(coord)
        #print(adj_mtx)

        label = np.array([self.label[imgname]])#.astype(np.float32)

        data = torch.from_numpy(img_group).float()
        label = torch.from_numpy(label).float()
        A = torch.from_numpy(adj_mtx).float()

        return data, label, imgname, A

    def __len__(self):
        return len(self.img_name)

if __name__ == '__main__':
    img_path, img_name, transform, is_training, label_path = '/home/fujun/datasets/iqa/CVIQ_Database/fov/', '001', None, False, '/home/fujun/datasets/iqa/CVIQ_Database/'
    i = get_dataset(True)

    dir = os.listdir('./database/cviqd_resize_imgtrain/001/')
    print(dir)