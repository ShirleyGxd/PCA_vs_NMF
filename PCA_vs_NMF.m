clear;clc;close all;

dataPath = 'CroppedYale_FrontFace\';
files = dir([dataPath '*.pgm']);

imgsize = 64;  
imgNum = length(files);
faceMatrix = zeros(imgsize*imgsize, imgNum);
for i = 1 : imgNum;
    imgName = [dataPath files(i).name];
    inImg = imread(imgName);

    smallImg = imresize(inImg, [imgsize, imgsize]);
     
    faceMatrix(:, i) = reshape(smallImg, [], 1);
%     faceMatrix(:, i) = smallImg(:);
end

% find the mean image and the mean-shifted input images
meanFace = mean(faceMatrix, 2);
shiftedFaceMatrix = faceMatrix - repmat(meanFace, 1, imgNum);
baseNum = 49;  

%% PCA
[pc, score, latent] = pca(shiftedFaceMatrix');
eigFaces = pc(:, 1:baseNum);
% figure; imshow(reshape(eigFaces(:,1), imgsize, imgsize), []);xlabel('eigFaces(1)');

%display the eigFaces
figure; 
for i = 1:baseNum
    subplot(7,7,i);
    imshow( reshape(imcomplement(eigFaces(: , i )) , imgsize , imgsize),[]);
end

%% NMF
% using the update rules to minimize the Euclidean distance
V = faceMatrix;

%%%%%%%%%%%%%%
% add your code here to decompose V to W and H
V=double(V);

%initial W and H
dim=size(V);
rand('seed',0);
W=10*rand(dim(1),baseNum);
rand('seed',1);
H=10*rand(baseNum,dim(2));

% iterations for non-negative matrix factorization
N=10000;
tol=0.01; % it need to be changed with the change of the boundary
err=zeros(N,1);
for i=1:N
    W0=W;
    W=W.*(V*H')./(W*(H*H'));
    H=H.*(W'*V)./(W'*W*H);
    
%   err(i)=norm(V-W*H,2); % when the boundary is ||V-W*H|| 
    err(i)=norm(W-W0,2);    % when the boundary is ||W(N)-W(N-1)|| 
    if(err(i)<tol)
        disp(['Iteration times are:' num2str(i)]);
        display(err(i));
        figure;plot(1:i,err(1:i));
        grid on;title('the error Convergence of NMF');
        break;
    end
end
if(err(i)>=tol)
    disp('N is not enough.');
    display(err(i));
    figure;plot(1:N,err);
    grid on; title('the error Convergence of NMF');
end

% restructed image by NMF
nmf=W*H;

%%%%%%%%%%%%%%
% figure; imshow(reshape(W(:, 1), imgsize, imgsize), []);xlabel('W(1)');

% display W
figure;
for i = 1:baseNum
    subplot(7,7,i);
    imshow( reshape(imcomplement(W(: , i )) , imgsize , imgsize),[]);
 end


% the conmparison of the original image_1 and the restructed image_1 get by NMF 
figure;
subplot(2,1,1);imshow( reshape(faceMatrix(: , 1) , imgsize , imgsize),[]);xlabel('ԭͼ');
subplot(2,1,2);imshow( reshape(nmf(: , 1) , imgsize , imgsize),[]);xlabel('�ع�ͼ by NMF');
