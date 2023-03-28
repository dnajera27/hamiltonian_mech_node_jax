% Builds and solves a simple least-squares problem using cvx

% Builds and solves a simple least-squares problem using cvx

clear all
clc
addpath('../')

%% ROM size
nr=1;
r_max = 6;          % maximum model size
%% Loading snapshot data
%FOM=load('David_FOM.mat');
%FOM = load('./ml-20230324-124511cubic_conserve-1-dof0064')
FOM = load('./ml-20230324-124511cubic_conserve-1-dof0064_small')
%FOM = out;
Q=(FOM.xw);     % Trajectory data
Qd=(FOM.xwdot);   % Velocity data
Qdd=(FOM.xwddot); % Acceleration data
%% Problem set-up
T       = FOM.t(end);      % final time
t      = FOM.t;     % time step
dt=t(2)-t(1);
Nt      = length(t); % Total number of time steps
n       = 64;      % Total number of nodes
start = 1;
%% Data-projection
Ntt=ceil(Nt/2);
% % compute POD basis for FOM data 
[Usvd,~,~] = svd(Q(:,start:Ntt),'econ');
V = Usvd(:,1:r_max);
%
eps=1e-12; % small eigenvalue to impose the positive definite constraint
%
% % %% calculate state and projection errors for different basis sizes
at=1;
proj_errors = zeros(r_max/nr,1);
state_errors = zeros(r_max/nr,1);
state_errors_test = zeros(r_max/nr,1);
%state_errors_test = zeros(r_max/nr,1);
Er1=zeros(at*Nt+1,r_max/nr);
for ii =1:1:r_max/nr         % loop through basis sizes
r=ii*nr
% Basis matrix for r-dimensional ROM
v = V(:,1:r);
%
Qr=(v'*Q(:,start:end));
Qdr=(v'*Qd(:,start:end));
Qddr=(v'*Qdd(:,start:end)); % found typo!

%Time-derivative data for Lagrangian operator inference
%[dQdt_2,d2Qdt2_2,ind_2] = ddt(Qhat,dt,'2c');
%[dQdt_4,d2Qdt2_4,ind_4] = ddt(Qhat,dt,'4c');
% Residual vector
%
%Qr=Qhat(:,ind_4); % Reduced snapshot data
%Qdr=dQdt_4;    % Reduced time-derivative data
%Qddr=d2Qdt2_4;    % Reduced double time-derivative data
%
%Khat_init=v'*K*v;
Khat_init=eye(r);
%Khat=rand(2*r,2*r);
Mhat=eye(r);
Chat=zeros(r);
%
for j=1:1
cvx_begin 
% Block for obtaining Mhat
variable Khat(r,r)  symmetric
minimize( norm( Qddr(:,1:end)'+Qr(:,1:end)'*Khat','fro' ) )
subject to
(Khat - eps*eye(r))==semidefinite(r)


cvx_end
% Khat_init=Khat;
%Khat=(Qr(1:r,1:end)'\(-Qddr(1:r,1:end)'*Mhat))';
end
%
%
% Projected initial conditions
qr1k=v'*Q(:,1);
qrk=v'*Q(:,2);
% Numerical integrating ROM using variational integrators
display(Nt)
XNhat= VI_ROM(qr1k,qrk,Mhat,Khat,dt,Nt,r);
Qrr=v*XNhat;   % Projecting back to FOM space
%
% State error
state_errors(ii) = state_errors(ii) + norm(Q(:,start:Ntt)-v*XNhat(:,start:Ntt),'fro')^1/norm(Q(:,start:Ntt),'fro')^1;
state_errors_test(ii) = state_errors_test(ii) + norm(Q(:,Ntt:Nt)-v*XNhat(:,Ntt:Nt),'fro')^1/norm(Q(:,Ntt:Nt),'fro')^1;
%
% State error
% tic;
% state_errors(ii) = state_errors(ii) + norm(Q(:,1:Ntt)-Qrr(:,1:Ntt),'fro')^1/Qnorm;
% toc;
%state_errors_test(ii) = state_errors_test(ii) + norm(Q(:,Ntt:at*Nt)-v*XNhat(:,Ntt:at*Nt),'fro')^1/norm(Q(:,Ntt:at*Nt),'fro')^1;
%
%This block is for the ROM Energy conservation evaluation% 
%Khat_int=v'*K*v;
%Mhat_int=v'*M*v;
%Vrk=(XNhat(:,2:end)-XNhat(:,1:end-1))/dt;
%Qrk=(XNhat(:,2:end)+ XNhat(:,1:end-1))/2;
%Ered1 =ener_red(Qrk,Vrk,Khat_int,Mhat_int,Nt); % Computing E(a)
%Er1(:,ii)=Ered1;

% This block is for the FOM Energy conservation evaluation%     
%tic;
%Vk=(Qrr(:,2:end)-Qrr(:,1:end-1))/dt;
%Qk=(Qrr(:,2:end)+Qrr(:,1:end-1))/2;
%Ered1 =ener(Qk,Vk,K,M,Nt); % Computing E(Va)
%Er1(:,ii)=Ered1;
%toc;
end

size(Qrr)

%save DOF6_ROM_ml-20230324-124511cubic_conserve-1-dof0064.mat Chat Mhat Khat Q Qd Qdd v t dt Qr Qdr Qddr Qrr XNhat Ntt
%%

figure
plot(XNhat')
hold on 
plot(Qr','k--')



dy = diff(XNhat')'/dt; % quick check, could improve differentiation 
y = XNhat(:, 2:end);
figure
plot(dy')
hold on
plot(Qdr', 'k--');

%% computing energy in reduced order space
[E_int] = ener_red(y, dy, Khat, Mhat, Ntt);
[E_red] = ener_red(Qr, Qdr, Khat, Mhat, Ntt);

figure
plot(E_int)
hold on
plot(E_red)
legend('Energy from integrated response', 'Energy from projected response');
ylim([10, 13])
title('Total Energy')

%%
% r_max = 100
% r = 100;
% V = Usvd(:,1:r_max);
% v = V(:,1:r);
% Ntt = 3000;
% %
% Qr=(v'*Q(:,start:Ntt));
% Qdr=(v'*Qd(:,start:Ntt));
% Qddr=(v'*Qdd(:,start:Ntt)); % found typo!
% 
% Q_ = v*Qr;
% 
% figure
% plot(Q(1,start:Ntt));
% hold on
% plot(Q_(1, 1:Ntt))

%% %% Variational integrator for the Lagrangian ROM
function X = VI_ROM(qr1k,qrk,Mhat,Khat,dt,Nt,n)
X=zeros(n,Nt+1);
X(:,1:2)=[qr1k,qrk];
M=(Mhat/dt) + (Khat*dt/4);
cond(M)
for i = 1:Nt
    b=(Mhat*(2*qrk-qr1k)/dt)-(Khat*dt*(2*qrk+qr1k)/4);
    xsol=M\b;
    X(:,i+2)=xsol;
    qr1k=qrk;
    qrk=xsol;
end
end
%% Function for computing the total energy E
function [E] = ener(Q,V,K,M,Nt)
for i=1:Nt+1 % loop over time points
   E(i,1)=0.5*(V(:,i)'*M*V(:,i)) + 0.5*(Q(:,i)'*K*Q(:,i));
end
end
%% Function for computing the total energy E
function [E] = ener_red(Q,V,Khat,Mhat,Nt)
for i=1:Nt+1 % loop over time points
   E(i,1)=0.5*(V(:,i)'*Mhat*V(:,i)) + 0.5*(Q(:,i)'*Khat*Q(:,i));
end
end