clc;
close all;
clear all;
tic;
% Image input
im_in = imread('flowersc.png');% Color image
size_temp = size(im_in);

% Text inputs
im_tx = imread('text.jpg');  % Text page 1
if( ~isequal(size_temp, size(im_tx)) )
    error('All Input Images must be equal in size');
end
size_temp = size(im_tx);
im_in_tx1(:,:,1) = im_tx(:,:,1);

im_tx = imread('black-letter-s-512.jpg');  % Text page 2
if( ~isequal(size_temp, size(im_tx)) )
    error('All Input Images must be equal in size');
end
size_temp = size(im_tx);
im_in_tx1(:,:,2) = im_tx(:,:,1);

im_tx = imread('black-letter-m-512.jpg');  % Text page 3
if( ~isequal(size_temp, size(im_tx)) )
    error('All Input Images must be equal in size');
end
size_temp = size(im_tx);
im_in_tx1(:,:,3) = im_tx(:,:,1);

%clear im_tx;
figure(1);
% Show inputs
title('Images to be encoded');
subplot(2,3,1);imshow(im_in_tx1(:,:,1));title({'Page 1'});
subplot(2,3,3);imshow(im_in_tx1(:,:,2));title('Page 2');
subplot(2,3,5);imshow(im_in_tx1(:,:,3));title('Page 3');

n= size(im_in_tx1(:,:,2),2)
k=300;
im_in_tx1(:,:,1) = circshift(im_in_tx1(:,:,1), [0, k])
im_in_tx1(:,:,2) = circshift(im_in_tx1(:,:,2), [0, k])
im_in_tx1(:,:,3) = circshift(im_in_tx1(:,:,3), [0, k])


imwrite(im_in_tx1(:,:,1),'1_embed.bmp');
imwrite(im_in_tx1(:,:,2),'2_embed.bmp');
imwrite(im_in_tx1(:,:,3),'3_embed.bmp');

figure(2);
imshow(im_in_tx1);

my_image = im2double(imread('2_embed.jpg'));
my_image = my_image(:,:,1);
% allocate space for thresholded image
image_thresholded = zeros(size(my_image));
% loop over all rows and columns
for ii=1:size(my_image,1)
    for jj=1:size(my_image,2)
        % get pixel value
        pixel=my_image(ii,jj);
        new_pixel = pixel;
          % check pixel value and assign new value
          if mod(ii,2)==0 & mod(jj,2)==0
              
          if pixel<0.5
              new_pixel=256;
          else
              new_pixel=0;
         
          end
          
          end
          % save new pixel value in thresholded image
          image_thresholded(ii,jj)=new_pixel;
         
      end
  end

%im_in_tx1(:,:,2)=image_thresholded(:,:,1)

%figure(11);
%imshow(image_thresholded);

my_image = im2double(imread('1_embed.jpg'));
my_image = my_image(:,:,1);
% allocate space for thresholded image
image_thresholded = zeros(size(my_image));
% loop over all rows and columns
for ii=1:size(my_image,1)
    for jj=1:size(my_image,2)
        % get pixel value
        pixel=my_image(ii,jj);
        new_pixel = pixel;
          % check pixel value and assign new value
          if mod(ii,2)~=0 & mod(jj,2)~=0
              
          if pixel<0.5
              new_pixel=256;
          else
              new_pixel=0;
         
          end
          
          end
          % save new pixel value in thresholded image
          image_thresholded(ii,jj)=new_pixel;
         
      end
end

%im_in_tx1(:,:,1)=image_thresholded

  my_image = im2double(imread('3_embed.jpg'));
my_image = my_image(:,:,1);
% allocate space for thresholded image
image_thresholded = zeros(size(my_image));
% loop over all rows and columns
for ii=1:size(my_image,1)
    for jj=1:size(my_image,2)
        % get pixel value
        pixel=my_image(ii,jj);
        new_pixel = pixel;
          % check pixel value and assign new value
          if mod(ii,2)~=0 & mod(jj,2)~=0
              
          if pixel<0.5
              new_pixel=256;
          else
              new_pixel=0;
          end
          
          end
          % save new pixel value in thresholded image
          image_thresholded(ii,jj)=new_pixel;
         
      end
end

%  im_in_tx1(:,:,3)=image_thresholded

 % figure(12);
%imshow(im_in_tx1);


% Hide Text images in Color image
im_wm = bitset(im_in,1,im_in_tx1)%bitshift(im_in_tx1,-7));

figure(3);subplot(1,3,1);
imshow(im_in);
title({'Color image';'(Before Encoding)'});
figure(3);subplot(1,3,2);
imshow(im_wm);
title({'Watermarked image';'(After Encoding)'});

%uncomment to add salt and pepper noise
%distorted = imnoise(im_wm,'salt & pepper',0.1);

%uncomment to add scale attack
%scale = 0.98;
%distorted = imresize(im_wm, scale); % Try varying the scale factor.

%uncomment to add rotation attack
%theta = 30;
%distorted = imrotate(im_wm,theta); % Try varying the angle, theta.

%uncomment for no attack
distorted=im_wm;



subplot(1,3,3);
imshow(distorted);
title({'Distorted Watermarked image';'(After Encoding)'});


im_wm=distorted
psnr_calc = distorted

% Write watermarked image after Encode
imwrite(distorted,'wm.tif');
distorted = imread('wm.tif');

ptsOriginal  = detectSURFFeatures(im_in(:,:,1));
ptsDistorted = detectSURFFeatures(distorted(:,:,1));

[featuresOriginal,   validPtsOriginal]  = extractFeatures(im_in(:,:,1),  ptsOriginal);
[featuresDistorted, validPtsDistorted]  = extractFeatures(distorted(:,:,1), ptsDistorted);

indexPairs = matchFeatures(featuresOriginal, featuresDistorted);

matchedOriginal  = validPtsOriginal(indexPairs(:,1));
matchedDistorted = validPtsDistorted(indexPairs(:,2));

[tform, inlierDistorted, inlierOriginal] = estimateGeometricTransform(...
    matchedDistorted, matchedOriginal, 'similarity');

Tinv  = tform.invert.T;

ss = Tinv(2,1);
sc = Tinv(1,1);
scale_recovered = sqrt(ss*ss + sc*sc)
theta_recovered = atan2(ss,sc)*180/pi

angle=(-theta_recovered)
ang=double(angle)
im_wm = imrotate(distorted,ang);

if scale_recovered<0.99 | scale_recovered > 1.01
scale=1/(scale_recovered)
im_wm= imresize(distorted, scale);
end

im_wm(:,:,1) = medfilt2(im_wm(:,:,1));
im_wm(:,:,2) = medfilt2(im_wm(:,:,2));
im_wm(:,:,3) = medfilt2(im_wm(:,:,3));

im_xtr1=bitget(im_wm,1) * 255;
 
PSF = fspecial('laplacian');
im_out_tx1(:,:,1) = medfilt2(edgetaper(im_xtr1(:,:,1),PSF));
im_out_tx1(:,:,2) = medfilt2(edgetaper(im_xtr1(:,:,2),PSF));
im_out_tx1(:,:,3) = medfilt2(edgetaper(im_xtr1(:,:,3),PSF));

diff=(size(im_out_tx1(:,:,1),1)-512)/2;

if ceil(diff)>2
im_out_tx(:,:,1)=imcrop(im_out_tx1(:,:,1),[diff diff 512 512]);
im_out_tx(:,:,2)=imcrop(im_out_tx1(:,:,2),[diff diff 512 512]);
im_out_tx(:,:,3)=imcrop(im_out_tx1(:,:,3),[diff diff 512 512]);

else
im_out_tx(:,:,1)=im_out_tx1(:,:,1);
im_out_tx(:,:,2)=im_out_tx1(:,:,2);
im_out_tx(:,:,3)=im_out_tx1(:,:,3);

end

my_image = im2double(im_out_tx(:,:,2));
my_image = my_image(:,:,1);
% allocate space for thresholded image
image_thresholded = zeros(size(my_image));
% loop over all rows and columns
for ii=1:size(my_image,1)
    for jj=1:size(my_image,2)
        % get pixel value
        pixel=my_image(ii,jj);
        new_pixel = pixel;
          % check pixel value and assign new value
          if mod(ii,2)==0 & mod(jj,2)==0
              
          if pixel>0.5
              new_pixel=256;
          else
              new_pixel=0;
         
          end
          
          end
          % save new pixel value in thresholded image
          image_thresholded(ii,jj)=new_pixel;
         
      end
  end

%im_out_tx(:,:,2)=image_thresholded

%figure(6);
%imshow(image_thresholded);

my_image = im2double(im_out_tx(:,:,1));
my_image = my_image(:,:,1);
% allocate space for thresholded image
image_thresholded = zeros(size(my_image));
% loop over all rows and columns
for ii=1:size(my_image,1)
    for jj=1:size(my_image,2)
        % get pixel value
        pixel=my_image(ii,jj);
        new_pixel = pixel;
          % check pixel value and assign new value
          if mod(ii,2)~=0 & mod(jj,2)~=0
              
          if pixel>0.5
              new_pixel=256;
          else
              new_pixel=0;
         
          end
          
          end
          % save new pixel value in thresholded image
          image_thresholded(ii,jj)=new_pixel;
         
      end
end

%im_out_tx(:,:,1)=image_thresholded

  my_image = im2double(im_out_tx(:,:,3));
my_image = my_image(:,:,1);
% allocate space for thresholded image
image_thresholded = zeros(size(my_image));
% loop over all rows and columns
for ii=1:size(my_image,1)
    for jj=1:size(my_image,2)
        % get pixel value
        pixel=my_image(ii,jj);
        new_pixel = pixel;
          % check pixel value and assign new value
          if mod(ii,2)~=0 & mod(jj,2)~=0
              
          if pixel>0.5
              new_pixel=256;
          else
              new_pixel=0;
         
          end
          
          end
          % save new pixel value in thresholded image
          image_thresholded(ii,jj)=new_pixel;
         
      end
end

 %im_out_tx(:,:,3)=image_thresholded

 

im_out_tx(:,:,1) = circshift(im_out_tx(:,:,1), [0,n-k])
im_out_tx(:,:,2) = circshift(im_out_tx(:,:,2), [0,n-k])
im_out_tx(:,:,3) = circshift(im_out_tx(:,:,3), [0,n-k])



toc;
figure(4);

subplot(2,3,1);imshow(im_out_tx(:,:,1));title({'Extracted Text';'Page 1'});
subplot(2,3,3);imshow(im_out_tx(:,:,2));title('Page 2');
subplot(2,3,5);imshow(im_out_tx(:,:,3));title('Page 3');

%figure(5);
%imshow(im_out_tx(:,:,1));
%figure(6);
%imshow(im_out_tx(:,:,2));
%figure(7);
%imshow(im_out_tx(:,:,3));

imwrite(im_out_tx(:,:,1),'1_recov.bmp');
imwrite(im_out_tx(:,:,2),'2_recov.bmp');
imwrite(im_out_tx(:,:,3),'3_recov.bmp');

[rows, columns]=size(psnr_calc);
im_in=imresize(im_in,[rows NaN])
im_f=im_in(:,:,1)
im_g=psnr_calc(:,:,1)
mse =sum((im_g(:)-im_f(:)).^2)/(rows * columns);

psnr_value1=10*log10(255*255/mse);

im_f=im_in(:,:,2)
im_g=psnr_calc(:,:,2)
mse =sum((im_g(:)-im_f(:)).^2)/(rows * columns);

psnr_value2=10*log10(255*255/mse);

im_f=im_in(:,:,3)
im_g=psnr_calc(:,:,3)
mse =sum((im_g(:)-im_f(:)).^2)/(rows * columns);

psnr_value3=10*log10(255*255/mse);

psnr_value=(psnr_value1+psnr_value2+psnr_value3)/3;

im_f=imread('1_embed.bmp');
im_g=imread('1_recov.bmp');
im_g=imresize(im_g,[512 NaN])
im_h=im_f.*im_g
im_j=im_f.*im_f
ncc1 =(sum(im_h(:)))/(sum(im_j(:)));
%nc1= corr2(double(im_f),double(im_g));

im_f=imread('2_embed.bmp');
im_g=imread('2_recov.bmp');
im_g=imresize(im_g,[512 NaN])
im_h=im_f.*im_g
im_j=im_f.*im_f
ncc2 =(sum(im_h(:)))/(sum(im_j(:)));

im_f=imread('3_embed.bmp');
im_g=imread('3_recov.bmp');
im_g=imresize(im_g,[512 NaN])
im_h=im_f.*im_g
im_j=im_f.*im_f
ncc3 =(sum(im_h(:)))/(sum(im_j(:))); 