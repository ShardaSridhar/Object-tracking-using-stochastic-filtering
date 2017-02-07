clear all;
close all;
clc;

files_current_dir=dir(strcat(pwd,'\img\*.jpg'));
files={files_current_dir.name};
duration=length(files);% No of images 
v = VideoWriter('UKF.avi');
open(v);


I=imread(strcat(pwd,'\img\0001.jpg'));% Initial observation
[m,n]=size(im2bw(I));
% R=redlikelihood(I);
x_actual=zeros(4,duration);
x_actual(:,1)=[0,0,48,80];

x_ukf = zeros(4,duration);
x_ukf(:,1) = x_actual(:,1);
y = zeros(2,duration);
dp = zeros(2,duration);

p_ukf = zeros(4,4);
p_ukf(1,1) = 0.00001;
p_ukf(2,2) = 0.00001;
p_ukf(3,3) = 0.00001;
p_ukf(4,4) = 0.00001;
% v = VideoWriter('Sharada_Sridhar_Coke_Video.avi');
% open(v);
delta=2;
sigma1=0.1;
sigma2=0.001;
Q = [sigma1*eye(2),zeros(2);zeros(2),sigma2*eye(2)];
g=dlmread('groundtruth_rect.txt');
y=g(:,1:2)';
for k=2:duration
    noise = Q*randn([4,duration]);
    F = [eye(2),delta*eye(2);zeros(2),eye(2)];
    x_actual(:,k) = F*x_actual(:,k-1)+noise(:,k);
    mat = chol(p_ukf);
    for j = 1:8
        if(j<=4)
            x_i(:,j) = x_ukf(:,k-1)+sqrt(4).*mat(:,j);
        else
            x_i(:,j) = x_ukf(:,k-1)-sqrt(4).*mat(:,j-4);
        end
        x_i(:,j) = F*x_i(:,j);
    end
    x_ukf(:,k) = mean(x_i,2);
    % To caluculate the value of P -- deviation of all vaues from the mean
    % value
    for i= 1:8
        if i == 1
            P_u = (x_i(:,i)-x_ukf(:,k))*(x_i(:,i)-x_ukf(:,k))';
        else
            P_u = P_u+(x_i(:,i)-x_ukf(:,k))*(x_i(:,i)-x_ukf(:,k))';
        end
    end
    P_u = (1/8).*P_u+Q;
    mat = chol(P_u);
    for n = 1:8
        if(n<=4)
            x_i(:,n) = x_ukf(:,k)+sqrt(4).*mat(:,n);
        else
            x_i(:,n) = x_ukf(:,k-1)-sqrt(4).*mat(:,n-4);
        end
        y_esti(:,n) = [x_i(1:2,n)]; 
    end
    y_estimated(:,k) = mean(y_esti,2);
    for m= 1:8
        if m == 1
            P_y = (y_esti(:,m)-y_estimated(:,k))*(y_esti(:,m)-y_estimated(:,k))';
            P_xy = (x_i(:,m)-x_ukf(:,k))*(y_esti(:,m)-y_estimated(:,k))';
        else
            P_y = P_y+(y_esti(:,m)-y_estimated(:,k))*(y_esti(:,m)-y_estimated(:,k))';
            P_xy = P_xy+(x_i(:,m)-x_ukf(:,k))*(y_esti(:,m)-y_estimated(:,k))';
        end
    end
    P_y = (1/8).*P_y;
    P_xy =(1/8).*P_xy;
    
    % To calculate gain
    K_k = P_xy*inv(P_y);
    x_ukf(:,k)= x_ukf(:,k)+K_k*(y(:,k)-y_estimated(:,k));
    d_1=y_estimated(1,k);
    d_2=y_estimated(2,k);
   
    if d_1<11
        d1=240;
    elseif d_1>469
        d1=240;
    end
    if d_2<11
        d_2=320;
    elseif d_2>629
        d_2=320;
    end
   
    d=y(:,k)';
    dp(:,k)=d';
    I= imread(strcat(pwd,'\img\',files{k}));
    I(d(2)-10:d(2)+10,d(1)-10:d(1)+10,1)=zeros(21);
    I(d(2)-10:d(2)+10,d(1)-10:d(1)+10,2)=zeros(21);
    I(d(2)-10:d(2)+10,d(1)-10:d(1)+10,3)=zeros(21);
    imagesc(I);
    fig=figure(1);
    f=getframe(fig);
    writeVideo(v,f);
    p_ukf = P_u;%K_k*P_y*K_k';
 
end

error=mean(mean(sqrt(y-y_estimated).^2))
close(v);
