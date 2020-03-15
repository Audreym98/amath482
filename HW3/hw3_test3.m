clear all; close all; clc; 
load('cam1_3.mat')
load('cam2_3.mat')
load('cam3_3.mat')
%%
%Play Videos
frames1 =size(vidFrames1_3);
frames2 =size(vidFrames2_3);
frames3 =size(vidFrames3_3);

%%
for k = 1:frames1(4)
    mov1(k).cdata = vidFrames1_3(:,:,:,k);
    mov1(k).colormap = [];
end

%Play video
width = 50;
filter = zeros(480,640); % size of frame
filter(300-1.5*width:1:300+3*width, 350-1.5*width:1:350+2*width) = 1; % black out around paint can


data1 = [];
for j=1:frames1(4)
    X=frame2im(mov1(j));
    X = double(rgb2gray(X));
    Xf = X.*filter;
    thresh = Xf > 250; % pick out bright points
    coordinates = find(thresh);
    [Y, X] = ind2sub(size(thresh),coordinates);
    
    data1 = [data1; mean(X), mean(Y)];
    % play filtered video
%      subplot(1,2,1)
%      imshow(uint8((thresh * 255))); drawnow
%      subplot(1,2,2)
%      imshow(uint8(Xf)); drawnow
end

%pause(5)
close all;
%%
for k = 1:frames2(4)
    mov2(k).cdata = vidFrames2_3(:,:,:,k);
    mov2(k).colormap = [];
end

filter = zeros(480,640);
filter(250-3*width:1:250+3.5*width, 290-2.5*width:1:290+2.5*width) = 1;

data2 = [];
%Play video
for j=1:frames2(4)
    X=frame2im(mov2(j));
    X = double(rgb2gray(X));
    Xf = X.*filter;
    thresh = Xf > 250;
    indeces = find(thresh);
    [Y, X] = ind2sub(size(thresh),indeces);
    data2 = [data2; mean(X), mean(Y)];
    
%       subplot(1,2,1)
%       imshow(uint8((thresh * 255))); drawnow
%       subplot(1,2,2)
%       imshow(uint8(Xf)); drawnow
end

%%
for k = 1:frames3(4)
    mov3(k).cdata = vidFrames3_3(:,:,:,k);
    mov3(k).colormap = [];
end

filter = zeros(480,640);
filter(250-2*width:1:250+2.5*width, 360-2.5*width:1:360+2.5*width) = 1;

data3 = [];
%Play video
for j=1:frames3(4)
    X=frame2im(mov3(j));
    X = double(rgb2gray(X));
    Xf = X.*filter;
    thresh = Xf > 247;
    indeces = find(thresh);
    [Y, X] = ind2sub(size(thresh),indeces);
    
    data3 = [data3; mean(X), mean(Y)];
    
%       subplot(1,2,1)
%       imshow(uint8((thresh * 255))); drawnow
%       subplot(1,2,2)
%       imshow(uint8(Xf)); drawnow
end

%% Synch frames based on min y value
[M,I] = min(data1(1:20,2));
data1  = data1(I:end,:);

[M,I] = min(data2(1:20,2));
data2  = data2(I:end,:);

[M,I] = min(data3(1:20,2));
data3  = data3(I:end,:);

%% Trim frames to min size
min = min([length(data1), length(data2), length(data3)]);
data1 = data1(1:min, :);
data2 = data2(1:min, :);
data3 = data3(1:min, :);

%% SVD
X = [data1';data2';data3'];
%alldata = alldata';

[m,n]=size(X); % compute data size
mn=mean(X,2); % compute mean for each row
X=X-repmat(mn,1,n); % subtract mean

[u,s,v]=svd(X); % perform the SVD
lambda=diag(s).^2; % produce diagonal variances
Y = u'*X;
%%
figure()
plot(1:6, lambda/sum(lambda), 'mo--', 'Linewidth', 2);
title("Test 3: Energy of Each Mode");
xlabel("Modes"); ylabel("Energy");
% First 4 captures most

figure()
subplot(3,1,1)
plot(1:6, lambda/sum(lambda), 'ro--', 'Linewidth', 2);
title("Test 3: Energy of Each Mode");
xlabel("Modes"); ylabel("Energy");
subplot(3,1,2)
plot(1:min, Y(1,:), 1:min, Y(2,:), 1:min, Y(3,:),1:min, Y(4,:),'m','Linewidth', 2)
ylabel("Displacement"); xlabel("Time"); 
title("Test 3: Projection Onto Principal Components");
legend("PC1", "PC2", "PC3", "PC4")
subplot(3,1,3)
plot(1:min, X(2,:),1:min, X(1,:), 'Linewidth', 2)
ylabel("Displacement"); xlabel("Time"); 
title("Test 3: Original displacement (cam 1)");
legend("Z", "X-Y")