clc;
clear;
close all;

%% -------------------------------
%  System Parameters
g  = 9.81;      % gravitational acceleration (m/s^2)

m1 = 8;         % mass of lower body / leg (kg)
m2 = 35;        % mass of torso / upper body (kg)

l1 = 0.9;       % length of lower link (m)
c1 = 0.45;      % COM of lower link from ankle (m)
c2 = 0.30;      % COM of upper link from hip (m)

I1 = 0.60;      % inertia of lower link about COM (kg m^2)
I2 = 1.20;      % inertia of upper link about COM (kg m^2)

r  = 0.50;      % point of external force application from hip (m)

%% -------------------------------
%  Linearized Matrices

% Useful constants
a = I1 + I2 + m1*c1^2 + m2*(l1^2 + c2^2);
b = m2*l1*c2;
d = I2 + m2*c2^2;

% Inertia matrix at upright equilibrium
M0 = [a + 2*b,  d + b;
      d + b,    d];

% Gravity (destabilizing stiffness) matrix
K = [ (m1*c1 + m2*l1 + m2*c2)*g,   m2*c2*g;
      m2*c2*g,                      m2*c2*g ];

% Input torque mapping
B = [1 -1;
     0  1];

% External force mapping
E0 = [l1 + r;
      r];

%% -------------------------------
%  State-Space Matrices

A = [ zeros(2), eye(2);
      M0\K,     zeros(2) ];

Bu = [ zeros(2,2);
       M0\B ];

Bf = [ zeros(2,1);
       M0\E0 ];

% Assume joint angles are measured
C = [1 0 0 0;
     0 1 0 0];

D = zeros(2,2);

%% -------------------------------
%  Stability Verification

eig_A = eig(A);

disp('------------------------------------');
disp('Eigenvalues of A:');
disp(eig_A);

%% -------------------------------
%  Controllability Verification
Co = ctrb(A, Bu);
rank_Co = rank(Co);

disp('------------------------------------');
disp('Controllability Matrix Rank:');
disp(rank_Co);

%% -------------------------------
%  Observability Verification

Ob = obsv(A, C);
rank_Ob = rank(Ob);

disp('------------------------------------');
disp('Observability Matrix Rank:');
disp(rank_Ob);

%% -------------------------------
disp('------------------------------------');
disp('A matrix = ');
disp(A);

disp('------------------------------------');
disp('Bu matrix = ');
disp(Bu);

disp('------------------------------------');
disp('Bf matrix = ');
disp(Bf);