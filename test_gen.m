%% Quick test: compile with generated C file and compare to original
clear; clc;
fprintf('=== Testing generated C implementation ===\n\n');

%% Compile generated version
fprintf('Compiling generated C version...\n');
% Delete any existing MEX
if exist('cr3bp_stt_time_derivatives_mex', 'file')
    delete(['cr3bp_stt_time_derivatives_mex.' mexext]);
end
% Single-step C-only compile (no need for two-step build!)
mex('-O', 'cr3bp_stt_time_derivatives_mex.c', 'cr3bp_stt_time_derivatives_gen.c');
fprintf('Compilation successful.\n\n');

%% Test parameters
mu = 0.012150585609624;
X0 = [0.8234; 0.0; 0.0; 0.0; -0.1263; 0.0];
Phi1_0 = eye(6);
Phi2_0 = zeros(6,6,6);
Phi3_0 = zeros(6,6,6,6);
Phi4_0 = zeros(6,6,6,6,6);
N = 10;

fprintf('Running MEX with N=%d...\n', N);
tic;
[Xders, Phi1ders, Phi2ders, Phi3ders, Phi4ders] = ...
    cr3bp_stt_time_derivatives_mex(X0, Phi1_0, Phi2_0, Phi3_0, Phi4_0, mu, N);
t_gen = toc;
fprintf('Generated version: %.4f ms\n\n', t_gen*1000);

%% Basic sanity checks
fprintf('--- Sanity checks ---\n');

% Check sizes
assert(all(size(Xders) == [6, N+1]), 'Xders size mismatch');
assert(all(size(Phi1ders) == [6, 6, N+1]), 'Phi1ders size mismatch');
assert(all(size(Phi2ders) == [6, 6, 6, N+1]), 'Phi2ders size mismatch');
assert(all(size(Phi3ders) == [6, 6, 6, 6, N+1]), 'Phi3ders size mismatch');
assert(all(size(Phi4ders) == [6, 6, 6, 6, 6, N+1]), 'Phi4ders size mismatch');
fprintf('  Output sizes: PASSED\n');

% Check X^(0) = X0
assert(max(abs(Xders(:,1) - X0)) < 1e-14, 'X0 mismatch');
fprintf('  X^(0) = X0: PASSED\n');

% Check Phi1^(0) = I
assert(max(max(abs(Phi1ders(:,:,1) - eye(6)))) < 1e-14, 'Phi1_0 mismatch');
fprintf('  Phi1^(0) = I: PASSED\n');

% Check X^(1) = f(X0) (equations of motion)
q1 = X0(1) + mu; q2 = X0(1) - 1 + mu;
r1 = sqrt(q1^2 + X0(2)^2 + X0(3)^2);
r2 = sqrt(q2^2 + X0(2)^2 + X0(3)^2);
ax = 2*X0(5) + X0(1) - (1-mu)*q1/r1^3 - mu*q2/r2^3;
ay = -2*X0(4) + X0(2) - (1-mu)*X0(2)/r1^3 - mu*X0(2)/r2^3;
az = -(1-mu)*X0(3)/r1^3 - mu*X0(3)/r2^3;
f_X0 = [X0(4); X0(5); X0(6); ax; ay; az];
err_f = max(abs(Xders(:,2) - f_X0));
fprintf('  X^(1) = f(X0) error: %.2e', err_f);
if err_f < 1e-12
    fprintf(' PASSED\n');
else
    fprintf(' FAILED\n');
end

% Check first derivative of Phi1: Phi1^(1) should be A(X0)*Phi1(0) = A(X0)
A_check = Phi1ders(:,:,2);  % This is Phi1^(1) = A(X0)
% Verify A(3,4) = 2 and A(4,3) = -2
fprintf('  A(4,5) = %.1f (expect 2.0): %s\n', A_check(4,5), ...
    tern(abs(A_check(4,5) - 2.0) < 1e-14, 'PASSED', 'FAILED'));
fprintf('  A(5,4) = %.1f (expect -2.0): %s\n', A_check(5,4), ...
    tern(abs(A_check(5,4) + 2.0) < 1e-14, 'PASSED', 'FAILED'));

% State propagation check: Taylor sum vs ode113
fprintf('\n--- Taylor propagation vs ODE113 ---\n');
dt = 0.001;
cr3bp_rhs = @(t, X) cr3bp_eom(X, mu);
opts = odeset('RelTol', 1e-13, 'AbsTol', 1e-15);
[~, Xode] = ode113(cr3bp_rhs, [0, dt], X0, opts);
X_ode = Xode(end, :)';

% Taylor sum: X(dt) = sum_{k=0}^{N} X^(k)/k! * dt^k = sum Xc[k] * dt^k
X_taylor = zeros(6,1);
for kk = 0:N
    X_taylor = X_taylor + Xders(:,kk+1) / factorial(kk) * dt^kk;
end
err_state = max(abs(X_taylor - X_ode));
fprintf('  State error at dt=%.4f: %.2e', dt, err_state);
if err_state < 1e-10
    fprintf(' PASSED\n');
else
    fprintf(' FAILED\n');
end

fprintf('\n=== All basic tests complete ===\n');

function out = tern(cond, a, b)
    if cond, out = a; else, out = b; end
end

function dXdt = cr3bp_eom(X, mu)
    x = X(1); y = X(2); z = X(3);
    vx = X(4); vy = X(5); vz = X(6);
    mu1 = 1 - mu;
    q1 = x + mu; q2 = x - 1 + mu;
    r1 = sqrt(q1^2 + y^2 + z^2);
    r2 = sqrt(q2^2 + y^2 + z^2);
    ax = 2*vy + x - mu1*q1/r1^3 - mu*q2/r2^3;
    ay = -2*vx + y - mu1*y/r1^3 - mu*y/r2^3;
    az = -mu1*z/r1^3 - mu*z/r2^3;
    dXdt = [vx; vy; vz; ax; ay; az];
end
