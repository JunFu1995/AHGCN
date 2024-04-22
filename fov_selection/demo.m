clear all;
clc;

% read images of dir 
imgPath = '/home/fujun/datasets/iqa/CVIQ_Database/CVIQ/';
imgDir = dir([imgPath  '*.png'])
savepath = '../fov/'
mkdir(savepath);
for i = 1:length(imgDir)
    imgname = imgDir(i).name
    sn = split(imgname,'.');
    sn = sn{1,1};
    
    img_dis_rgb=imread([imgPath imgname]);
    [phi theta]=select_points(img_dis_rgb);
    spoint_radian = [phi' theta'];
    mkdir([savepath sn]);
    img_dis_rgb=imresize(img_dis_rgb,[512 1024]);
    im2fov(img_dis_rgb,spoint_radian, sn,[savepath  sn '/']);
    coord = [sn '_coord.mat'];
    sp = [savepath  sn '/' coord]
    save(sp, 'spoint_radian');
end