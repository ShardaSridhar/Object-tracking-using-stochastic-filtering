%% Tracks the coke can in the video sequence based on the condensation algorithm
clear all;
close all;

files_current_dir=dir(strcat(pwd,'\Coke\*.jpg'));
files={files_current_dir.name};
duration=length(files);% No of images 

I=imread(strcat(pwd,'\Coke\0001.jpg'));% Initial observation
[m,n]=size(im2bw(I));
R=redlikelihood(I);% contains the amount of redness likely to each pixel
R1=reshape(R,1,m*n);

N=1000;% Number of particles used
[p,index_p]=datasample(R1,N,'Replace',false);% Draw N particles uniformly 
p=p/sum(p); % Get the paricles to form a proper pmf by normalizing them - posterior density
[m1,n1] = ind2sub([m,n],index_p);
x= [m1;n1;zeros(1,N);zeros(1,N)];

v = VideoWriter('Sharada_Sridhar_Coke_Video.avi');
open(v);

delta=2;
sigma1=0.01;
sigma2=2;

%Start the particle filtering
for k=1:duration
    
    % Draw particles from the new posterior weighted according to
    % likelihood
    new_indices = zeros(1,N);
    indices = new_indices;
    p_new = cumsum(p);
     
     for i = 1:N
        u = rand;
        most_likely_indices = find(p_new>=u);
        new_indices(i) = index_p(most_likely_indices(1));
        indices(i) = most_likely_indices(1);
     end	
   
    new_likelihood = p(indices);
    x_new = x(:,indices);
    clear new_indices indices;
    
    % Define the system dynamics model
    x_update = round([eye(2),delta*eye(2);zeros(2),eye(2)]*x_new+[sigma1*eye(2),zeros(2);zeros(2),sigma2*eye(2)]*(mvnrnd(zeros(length(p),4),eye(4)))');
    [~,n2] = find(x_update(1,:)<=0);[~,n3] = find( x_update(2,:)<=0);[~,n4] = find(x_update(1,:)> m);[~,n5] = find( x_update(2,:)> n);
    x_update(1,n2) = 1;x_update(2,n3) = 1;x_update(1,n4) = m;x_update(2,n5) = n;
    
    new_likelihood = new_likelihood/sum(new_likelihood);
    plotstep(x_update,I,new_likelihood,k,30);
   
    
    fig = figure(1);
    f = getframe(fig);
    writeVideo(v,f);
    
    index_p= sub2ind(size(R),x_update(1,:),x_update(2,:));
    x =x_update;    
    
    clear R R1 p x_new new_likelihood m1 n1;
    % Update the observation
    I= imread(strcat(pwd,'\Coke\',files{k}));
    R = redlikelihood(I);
    R1 = reshape(R,1,m*n);
    p = R1(index_p);
    p= p/sum(p);
end
close(v);