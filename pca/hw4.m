function hw4

load teapots.mat

mean_val = mean(teapotImages, 1);
cov_matrix = cov(teapotImages);
[vector, val] = eig(cov_matrix);
index = [1898;1899;1900];
for i = 1:1:3
    subplot(1,3,i);
    imagesc(reshape(vector(:,index(i)),38,50));
    colormap gray;
    hold on;
end

for i = 1:10
    subplot(2,5,i);
    imagesc(reshape(teapotImages(10*i,:),38,50));
    colormap gray;
    hold on;
end


for i=1:10
    new(:,i)=(mean_val'+ i * vector(:,index(1)) + 1/3 * vector(:, index(2))+(2/3 - i) * vector(:,index(3)));
end



for i = 1:10
    subplot(2,5,i);
    imagesc(reshape(new(:,i),38,50));
    colormap gray;
    hold on;
end

    