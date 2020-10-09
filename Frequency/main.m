im_in = imread('Lenna.png');
key=3;

im_in_r=im_in(:,:,1);
im_in_g=im_in(:,:,2);
im_in_b=im_in(:,:,3);

im_tx = imread('black-letter-i-512.jpg');
im_tx=imresize(((im_tx)),[32 32]);
wm_tx(:,:,1) = im_tx(:,:,1);
wtmark(im_in_r,im_tx,'out1.jpg');

im_tx = imread('black-letter-a-512.jpg');
im_tx=imresize(((im_tx)),[32 32]);
wm_tx(:,:,2) = im_tx(:,:,1);
wtmark(im_in_g,im_tx,'out2.jpg');

im_tx = imread('black-letter-m-512.jpg');
im_tx=imresize(((im_tx)),[32 32]);
wm_tx(:,:,3) = im_tx(:,:,1);
wtmark(im_in_b,im_tx,'out3.jpg');

figure(1);
imshow(wm_tx);

im_tx = imread('out1.jpg');
im_wm1(:,:,1)=im_tx(:,:,1);

im_tx = imread('out2.jpg');
im_wm1(:,:,2)=im_tx(:,:,1);

im_tx = imread('out3.jpg');
im_wm1(:,:,3)=im_tx(:,:,1);


%im_wm1(:,:,1)=medfilt2(im_wm1(:,:,1),[3 3]);
%im_wm1(:,:,2)=medfilt2(im_wm1(:,:,2),[3 3]);
%im_wm1(:,:,3)=medfilt2(im_wm1(:,:,3),[3 3]);

%im_wm1(:,:,1)=histeq(im_wm1(:,:,1));
%im_wm1(:,:,2)=histeq(im_wm1(:,:,2));
%im_wm1(:,:,3)=histeq(im_wm1(:,:,3));

%im_wm1(:,:,1) = imadjust(im_wm1(:,:,1),[],[],2);
%im_wm1(:,:,2) = imadjust(im_wm1(:,:,2),[],[],2);
%im_wm1(:,:,3) = imadjust(im_wm1(:,:,3),[],[],2);

%im_wm1=imsharpen(im_wm1);

%im_wm1 = imnoise(im_wm1,'salt & pepper',0.03);

 %im_wm1 = imnoise(im_wm1,'gaussian',0,0.01);
 
%o = fspecial('gaussian',[3 3],0.5);
%o = fspecial('average',[2 2]);
%im_wm1 = imfilter(im_wm1,o,'replicate');

K=im_wm1(:,:,3)
K(1:64,1:64)=0;
%im_wm1(:,:,3)=K;

K=im_wm1(:,:,2);
K(1:64,1:64)=0;
%im_wm1(:,:,2)=K;

K=im_wm1(:,:,1);
K(1:64,1:64)=0;
%im_wm1(:,:,1)=K;

figure(2);
imshow(im_wm1);

[mssim, ssim_map] = ssim(im_in, im_wm1);

    exwmark(im_wm1(:,:,1));
    
    im_f=imread('black-letter-i-512.jpg');
    im_f=imresize(((im_f)),[32 32]);
    im_f=im_f(:,:,1);
    im_f=double(im_f);
    im_g=imread('wex.jpg');
    im_g=double(im_g);
    mean_f=sum(im_f(:))/(32*32);
    mean_g=sum(im_g(:))/(32*32);
    
    p1=0;
    for i=1:32
        for j=1:32
            p1=p1+(im_f(i,j)-mean_f)*(im_g(i,j)-mean_g)
        end
    end
    p2=0;
    for i=1:32
        for j=1:32
            p2=p2+(im_f(i,j)-mean_f)*(im_f(i,j)-mean_f)
        end
    end
    p3=0;
    for i=1:32
        for j=1:32
            p3=p3+(im_g(i,j)-mean_g)*(im_g(i,j)-mean_g)
        end
    end
    
  %  ncc13= p1/(sqrt(p2)*sqrt(p3));
    
    im_h=im_f.*im_g
    ncc1 =(sum(im_h(:)))/(sum((im_f(:)).^2));
    
     c = normxcorr2(im_g,im_f);
    [maxCorrValue1, maxIndex] = max(abs(c(:)));
   
    im_g=imresize(((im_g)),[64 64])
    figure(3);
    imshow(im_g);
    
    
    exwmark(im_wm1(:,:,2));
    
    im_f=imread('black-letter-a-512.jpg');
    im_f=imresize(((im_f)),[32 32]);
    im_f=im_f(:,:,1);
    im_f=double(im_f);
    im_g=imread('wex.jpg');
    im_g=double(im_g);
    mean_f=sum(im_f(:))/(32*32);
    mean_g=sum(im_g(:))/(32*32);
    
    p1=0;
    for i=1:32
        for j=1:32
            p1=p1+(im_f(i,j)-mean_f)*(im_g(i,j)-mean_g)
        end
    end
    p2=0;
    for i=1:32
        for j=1:32
            p2=p2+(im_f(i,j)-mean_f)*(im_f(i,j)-mean_f)
        end
    end
    p3=0;
    for i=1:32
        for j=1:32
            p3=p3+(im_g(i,j)-mean_g)*(im_g(i,j)-mean_g)
        end
    end
    
   % ncc23= p1/(sqrt(p2)*sqrt(p3));
   
    im_f=imread('black-letter-a-512.jpg');
    im_f=imresize(((im_f)),[32 32]);
    im_f=im_f(:,:,1);
    im_g=imread('wex.jpg');
    im_h=im_f.*im_g
    ncc2 =(sum(im_h(:)))/(sum((im_f(:)).^2));
    
    c = normxcorr2(im_g,im_f);
    [maxCorrValue2, maxIndex] = max(abs(c(:)));
    
    im_g=imresize(((im_g)),[64 64])
    figure(4);
    imshow(im_g)
   
    exwmark(im_wm1(:,:,3));
    
   im_f=imread('black-letter-m-512.jpg');
    im_f=imresize(((im_f)),[32 32]);
    im_f=im_f(:,:,1);
    im_f=double(im_f);
    im_g=imread('wex.jpg');
    im_g=double(im_g);
    mean_f=sum(im_f(:))/(32*32);
    mean_g=sum(im_g(:))/(32*32);
    
    p1=0;
    for i=1:32
        for j=1:32
            p1=p1+(im_f(i,j)-mean_f)*(im_g(i,j)-mean_g)
        end
    end
    p2=0;
    for i=1:32
        for j=1:32
            p2=p2+(im_f(i,j)-mean_f)*(im_f(i,j)-mean_f)
        end
    end
    p3=0;
    for i=1:32
        for j=1:32
            p3=p3+(im_g(i,j)-mean_g)*(im_g(i,j)-mean_g)
        end
    end
    
  %  ncc33= p1/(sqrt(p2)*sqrt(p3));
    
    im_f=imread('black-letter-m-512.jpg');
    im_f=imresize(((im_f)),[32 32]);
    im_f=im_f(:,:,1);
    im_g=imread('wex.jpg');
    im_h=im_f.*im_g
    ncc3 =(sum(im_h(:)))/(sum((im_f(:)).^2));
    
    im_g=imresize(((im_g)),[64 64])
    figure(5);
    imshow(im_g)
   
[rows, columns]=size(im_wm1);
im_f=im_in(:,:,1)
im_g=im_wm1(:,:,1)
mse =sum((im_g(:)-im_f(:)).^2)/(rows * columns);

psnr_value1=10*log10(255*255/mse);

im_f=im_in(:,:,2)
im_g=im_wm1(:,:,2)
mse =sum((im_g(:)-im_f(:)).^2)/(rows * columns);

psnr_value2=10*log10(255*255/mse);

im_f=im_in(:,:,3)
im_g=im_wm1(:,:,3)
mse =sum((im_g(:)-im_f(:)).^2)/(rows * columns);

psnr_value3=10*log10(255*255/mse);

psnr_value=(psnr_value1+psnr_value2+psnr_value3)/3;