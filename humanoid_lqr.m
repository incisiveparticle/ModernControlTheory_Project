clc;
clear;
close all;

%% 
%  System Parameters
g  = 9.81;      % gravitational acceleration (m/s^2)

m1 = 8;         % lower body / leg mass (kg)
m2 = 35;        % upper body / torso mass (kg)

l1 = 0.9;       % lower link length (m)
c1 = 0.45;      % COM of lower link from ankle (m)
c2 = 0.30;      % COM of upper link from hip (m)

I1 = 0.60;      % lower link inertia about COM (kg m^2)
I2 = 1.20;      % upper link inertia about COM (kg m^2)

r  = 0.50;      % force application point from hip (m)

%% 
%  Linearized Matrices
a = I1 + I2 + m1*c1^2 + m2*(l1^2 + c2^2);
b = m2*l1*c2;
d = I2 + m2*c2^2;

M0 = [a + 2*b,  d + b;
      d + b,    d];

Kg = [ (m1*c1 + m2*l1 + m2*c2)*g,   m2*c2*g;
       m2*c2*g,                      m2*c2*g ];

B = [1 -1;
     0  1];

E0 = [l1 + r;
      r];

%% 
%  State-Space Matrices

A = [ zeros(2), eye(2);
      M0\Kg,    zeros(2) ];

Bu = [ zeros(2,2);
       M0\B ];

Bf = [ zeros(2,1);
       M0\E0 ];

C = [1 0 0 0;
     0 1 0 0];

%% 
%  Open-loop Properties

eig_A = eig(A);
Co = ctrb(A, Bu);
Ob = obsv(A, C);

disp('------------------------------------');
disp('Eigenvalues of A:');
disp(eig_A);

disp('------------------------------------');
disp('Controllability Rank:');
disp(rank(Co));

disp('------------------------------------');
disp('Observability Rank:');
disp(rank(Ob));

%%
%  LQR Design

theta1_max  = 5*pi/180;
theta2_max  = 5*pi/180;
dtheta1_max = 30*pi/180;
dtheta2_max = 30*pi/180;

Q = diag([1/theta1_max^2, ...
          1/theta2_max^2, ...
          1/dtheta1_max^2, ...
          1/dtheta2_max^2]);

R = diag([1, 1]);   % tuned control penalty

Klqr = lqr(A, Bu, Q, R);
Acl  = A - Bu*Klqr;

eig_Acl = eig(Acl);

disp('------------------------------------');
disp('LQR Gain Matrix Klqr = ');
disp(Klqr);

disp('------------------------------------');
disp('Closed-loop Eigenvalues = ');
disp(eig_Acl);

if all(real(eig_Acl) < 0)
    disp('Closed-loop system is asymptotically stable.');
else
    disp('Closed-loop system is NOT asymptotically stable.');
end

%%
%  Save variables for simulation

save('humanoid_lqr_data.mat', 'A', 'Bu', 'Bf', 'Acl', 'Klqr');


format long g

disp('==================== A ====================');
disp(A);

disp('==================== Acl ====================');
disp(Acl);

disp('==================== Bf ====================');
disp(Bf);

disp('==================== Klqr ====================');
disp(Klqr);