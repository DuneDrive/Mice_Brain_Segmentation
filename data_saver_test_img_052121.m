fid = fopen("/Users/hsm/Downloads/Mouse-Segmentation-master/my_img_paths.txt");
% Read all lines & collect in cell array
txt = textscan(fid,'%s','delimiter','\n');
path = "/Users/hsm/Downloads/Mouse-Segmentation-master/";
 
 
for i = 1:length(txt{1,1})
img = char(txt{1,1}(i));
img_path = strcat(path, img);
img_volume = niftiread(img_path);
img_rot = permute(img_volume, [1,3,2]);

if size(img_rot,1) == 200
    img_rot = img_rot(11:end-10,:,:);
end

full_data(i,:,:,:,1) = img_rot;


size(img_rot)

end


%%







niftiwrite(full_data,'mouse_full_data_052121');

 