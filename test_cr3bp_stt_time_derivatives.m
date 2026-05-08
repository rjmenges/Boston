%==========================================================================
% test_cr3bp_stt_time_derivatives.m
%
% Comprehensive test suite for the cr3bp_stt_time_derivatives_mex function.
%
% TESTS:
%   A. State-only Taylor propagation vs ode113
%   B. STM Taylor propagation vs ode113
%   C. Initial derivative (Phi1^(1) = A(X0)) verification
%   D. Symmetry of Phi2, Phi3, Phi4 derivatives
%   E. Finite-difference verification of Phi1, Phi2 (and optionally Phi3)
%   F. MEX interface sizing and indexing checks
%==========================================================================

clear; clc;
fprintf('=== CR3BP STT Time Derivatives Test Suite ===\n\n');

%% Compile MEX if needed
if ~exist('cr3bp_stt_time_derivatives_mex', 'file')
    fprintf('Compiling MEX file...\n');
    % Two-step build: compile the C++ engine to an object file first,
    % then link the C gateway against it. This prevents mex from using
    % clang++ for the .c file (which would trigger the new C++ MEX API).
    fprintf('  Step 1: Compiling C++ engine...\n');
    mex('-c', '-O', 'cr3bp_stt_time_derivatives.cpp');
    fprintf('  Step 2: Linking MEX gateway...\n');
    obj_ext = mexext; % e.g. 'mexmaca64'
    % The object file name depends on OS
    if ispc
        obj_file = 'cr3bp_stt_time_derivatives.obj';
    else
        obj_file = 'cr3bp_stt_time_derivatives.o';
    end
    mex('-O', 'cr3bp_stt_time_derivatives_mex.c', obj_file);
    % Clean up object file
    if exist(obj_file, 'file'), delete(obj_file); end
    fprintf('Compilation successful.\n\n');
end

%% Common setup
% Earth-Moon system mass parameter
mu = 0.012150585609624;

% L1 Lyapunov orbit initial condition (approximate)
% This is a typical halo-orbit-like IC near Earth-Moon L1
x0  =  0.8234;
y0  =  0.0;
z0  =  0.0;
vx0 =  0.0;
vy0 = -0.1263;
vz0 =  0.0;
X0 = [x0; y0; z0; vx0; vy0; vz0];

% Standard initial conditions for STTs
Phi1_0 = eye(6);
Phi2_0 = zeros(6,6,6);
Phi3_0 = zeros(6,6,6,6);
Phi4_0 = zeros(6,6,6,6,6);

N = 10;  % derivative order for most tests
dt = 1e-3; % small time step for propagation tests

pass_count = 0;
fail_count = 0;

%% ======================================================================
%  TEST F: MEX interface sizes and basic functionality
%  ======================================================================
fprintf('--- TEST F: MEX Interface ---\n');
try
    [Xders, Phi1ders, Phi2ders, Phi3ders, Phi4ders] = ...
        cr3bp_stt_time_derivatives_mex(X0, Phi1_0, Phi2_0, Phi3_0, ...
                                        Phi4_0, mu, N);

    % Check output sizes
    assert(all(size(Xders) == [6, N+1]), 'Xders size mismatch');
    assert(all(size(Phi1ders) == [6, 6, N+1]), 'Phi1ders size mismatch');
    assert(all(size(Phi2ders) == [6, 6, 6, N+1]), 'Phi2ders size mismatch');
    assert(all(size(Phi3ders) == [6, 6, 6, 6, N+1]), 'Phi3ders size mismatch');
    assert(all(size(Phi4ders) == [6, 6, 6, 6, 6, N+1]), 'Phi4ders size mismatch');

    % Check k=0 slice is the initial condition
    assert(norm(Xders(:,1) - X0) < 1e-14, 'Xders(:,1) should be X0');
    assert(norm(Phi1ders(:,:,1) - Phi1_0) < 1e-14, 'Phi1ders(:,:,1) should be Phi1_0');

    fprintf('  Output sizes: PASS\n');
    fprintf('  Initial conditions: PASS\n');
    pass_count = pass_count + 2;
catch ME
    fprintf('  FAIL: %s\n', ME.message);
    fail_count = fail_count + 1;
end

% Test N=0
try
    [Xd0, P1d0, P2d0, P3d0, P4d0] = ...
        cr3bp_stt_time_derivatives_mex(X0, Phi1_0, Phi2_0, Phi3_0, Phi4_0, mu, 0);
    assert(all(size(Xd0) == [6,1]), 'N=0 Xders size');
    assert(norm(Xd0 - X0) < 1e-14, 'N=0 Xders value');
    fprintf('  N=0 case: PASS\n');
    pass_count = pass_count + 1;
catch ME
    fprintf('  N=0 case FAIL: %s\n', ME.message);
    fail_count = fail_count + 1;
end

fprintf('\n');

%% ======================================================================
%  TEST C: Initial derivative Phi1^(1) = A(X0)
%  ======================================================================
fprintf('--- TEST C: Initial Derivative (Phi1dot = A*Phi1) ---\n');

% Build the CR3BP Jacobian at X0
A_x0 = cr3bp_jacobian(X0, mu);

% Phi1^(1) should equal A(X0) * Phi1_0 = A(X0) * I = A(X0)
Phi1_dot_mex = Phi1ders(:,:,2);  % k=1 raw derivative

err_A = norm(Phi1_dot_mex - A_x0, 'fro');
fprintf('  ||Phi1^(1) - A(X0)||_F = %.3e\n', err_A);

if err_A < 1e-12
    fprintf('  PASS\n');
    pass_count = pass_count + 1;
else
    fprintf('  FAIL\n');
    fail_count = fail_count + 1;
end

% Also check Xders(:,2) = f(X0)
f_x0 = cr3bp_eom(0, X0, mu);
err_f = norm(Xders(:,2) - f_x0);
fprintf('  ||X^(1) - f(X0)|| = %.3e\n', err_f);
if err_f < 1e-12
    fprintf('  PASS\n');
    pass_count = pass_count + 1;
else
    fprintf('  FAIL\n');
    fail_count = fail_count + 1;
end
fprintf('\n');

%% ======================================================================
%  TEST A: State-only Taylor propagation vs ode113
%  ======================================================================
fprintf('--- TEST A: State Taylor Propagation vs ode113 ---\n');

% ODE113 reference
opts = odeset('RelTol', 1e-14, 'AbsTol', 1e-14);
[~, Xref_sol] = ode113(@(t,X) cr3bp_eom(t,X,mu), [0, dt], X0, opts);
Xref = Xref_sol(end,:)';

% Taylor propagation using raw derivatives
Xtaylor = zeros(6,1);
for k = 0:N
    Xtaylor = Xtaylor + Xders(:,k+1) * dt^k / factorial(k);
end

err_state = norm(Xtaylor - Xref);
fprintf('  dt = %.1e,  N = %d\n', dt, N);
fprintf('  ||X_taylor - X_ode113|| = %.3e\n', err_state);

if err_state < 1e-12
    fprintf('  PASS\n');
    pass_count = pass_count + 1;
else
    fprintf('  FAIL (tolerance 1e-12)\n');
    fail_count = fail_count + 1;
end

% Test convergence with increasing N
fprintf('  Convergence with N:\n');
for Ntest = [2, 4, 6, 8, 10]
    [Xd_t, ~, ~, ~, ~] = cr3bp_stt_time_derivatives_mex(...
        X0, Phi1_0, Phi2_0, Phi3_0, Phi4_0, mu, Ntest);
    Xt = zeros(6,1);
    for k = 0:Ntest
        Xt = Xt + Xd_t(:,k+1) * dt^k / factorial(k);
    end
    fprintf('    N=%2d: error = %.3e\n', Ntest, norm(Xt - Xref));
end
fprintf('\n');

%% ======================================================================
%  TEST B: STM Taylor propagation vs ode113
%  ======================================================================
fprintf('--- TEST B: STM Taylor Propagation vs ode113 ---\n');

% Propagate state + STM with ode113
Y0 = [X0; reshape(eye(6), 36, 1)];
[~, Yref_sol] = ode113(@(t,Y) cr3bp_stm_eom(t,Y,mu), [0, dt], Y0, opts);
Yref = Yref_sol(end,:)';
Phi1_ref = reshape(Yref(7:42), 6, 6);

% Taylor propagation of Phi1
Phi1_taylor = zeros(6,6);
for k = 0:N
    Phi1_taylor = Phi1_taylor + Phi1ders(:,:,k+1) * dt^k / factorial(k);
end

err_stm = norm(Phi1_taylor - Phi1_ref, 'fro');
fprintf('  dt = %.1e,  N = %d\n', dt, N);
fprintf('  ||Phi1_taylor - Phi1_ode113||_F = %.3e\n', err_stm);

if err_stm < 1e-11
    fprintf('  PASS\n');
    pass_count = pass_count + 1;
else
    fprintf('  FAIL (tolerance 1e-11)\n');
    fail_count = fail_count + 1;
end
fprintf('\n');

%% ======================================================================
%  TEST D: Symmetry of higher-order STT derivatives
%  ======================================================================
fprintf('--- TEST D: Symmetry Tests ---\n');

% Phi2 should be symmetric in indices a,b (2nd and 3rd indices)
max_asym_2 = 0;
for k = 1:N+1
    for i = 1:6
        for a = 1:6
            for b = a+1:6
                diff = abs(Phi2ders(i,a,b,k) - Phi2ders(i,b,a,k));
                max_asym_2 = max(max_asym_2, diff);
            end
        end
    end
end
fprintf('  Phi2 max asymmetry in (a,b): %.3e\n', max_asym_2);
if max_asym_2 < 1e-12
    fprintf('  Phi2 symmetry: PASS\n');
    pass_count = pass_count + 1;
else
    fprintf('  Phi2 symmetry: FAIL\n');
    fail_count = fail_count + 1;
end

% Phi3 should be symmetric under permutations of (a,b,c)
max_asym_3 = 0;
for k = 1:N+1
    for i = 1:6
        for a = 1:6
            for b = 1:6
                for c = 1:6
                    vals = [Phi3ders(i,a,b,c,k), Phi3ders(i,a,c,b,k), ...
                            Phi3ders(i,b,a,c,k), Phi3ders(i,b,c,a,k), ...
                            Phi3ders(i,c,a,b,k), Phi3ders(i,c,b,a,k)];
                    max_asym_3 = max(max_asym_3, max(vals) - min(vals));
                end
            end
        end
    end
end
fprintf('  Phi3 max asymmetry in (a,b,c): %.3e\n', max_asym_3);
if max_asym_3 < 1e-10
    fprintf('  Phi3 symmetry: PASS\n');
    pass_count = pass_count + 1;
else
    fprintf('  Phi3 symmetry: FAIL\n');
    fail_count = fail_count + 1;
end

% Phi4: check a few symmetry relations
max_asym_4 = 0;
for k = 1:min(3, N+1)  % only check first few orders (expensive otherwise)
    for i = 1:6
        for a = 1:3  % subset for speed
            for b = 1:3
                for c = 1:3
                    for d = 1:3
                        v1 = Phi4ders(i,a,b,c,d,k);
                        v2 = Phi4ders(i,b,a,c,d,k);
                        v3 = Phi4ders(i,a,b,d,c,k);
                        v4 = Phi4ders(i,d,c,b,a,k);
                        vals = [v1, v2, v3, v4];
                        max_asym_4 = max(max_asym_4, max(vals) - min(vals));
                    end
                end
            end
        end
    end
end
fprintf('  Phi4 max asymmetry (sample): %.3e\n', max_asym_4);
if max_asym_4 < 1e-9
    fprintf('  Phi4 symmetry: PASS\n');
    pass_count = pass_count + 1;
else
    fprintf('  Phi4 symmetry: FAIL\n');
    fail_count = fail_count + 1;
end
fprintf('\n');

%% ======================================================================
%  TEST E: Finite-difference verification of Phi1 and Phi2
%  ======================================================================
fprintf('--- TEST E: Finite-Difference Verification ---\n');

dt_fd = 1e-3;  % propagation step
eps_fd = 1e-6; % finite-difference step

% Get Taylor derivatives at X0
Nfd = 8;
[Xd, P1d, P2d, P3d, P4d] = cr3bp_stt_time_derivatives_mex(...
    X0, Phi1_0, Phi2_0, Phi3_0, Phi4_0, mu, Nfd);

% Taylor-propagate the flow map
Xprop = zeros(6,1);
Phi1_prop = zeros(6,6);
Phi2_prop = zeros(6,6,6);
for k = 0:Nfd
    fk = dt_fd^k / factorial(k);
    Xprop = Xprop + Xd(:,k+1) * fk;
    Phi1_prop = Phi1_prop + P1d(:,:,k+1) * fk;
    Phi2_prop = Phi2_prop + P2d(:,:,:,k+1) * fk;
end

% --- Phi1 via finite differences ---
Phi1_fd = zeros(6,6);
for a = 1:6
    dX = zeros(6,1);
    dX(a) = eps_fd;

    % Forward
    X_plus = X0 + dX;
    [Xd_p, ~, ~, ~, ~] = cr3bp_stt_time_derivatives_mex(...
        X_plus, Phi1_0, Phi2_0, Phi3_0, Phi4_0, mu, Nfd);
    Xprop_p = zeros(6,1);
    for k = 0:Nfd
        Xprop_p = Xprop_p + Xd_p(:,k+1) * dt_fd^k / factorial(k);
    end

    % Backward
    X_minus = X0 - dX;
    [Xd_m, ~, ~, ~, ~] = cr3bp_stt_time_derivatives_mex(...
        X_minus, Phi1_0, Phi2_0, Phi3_0, Phi4_0, mu, Nfd);
    Xprop_m = zeros(6,1);
    for k = 0:Nfd
        Xprop_m = Xprop_m + Xd_m(:,k+1) * dt_fd^k / factorial(k);
    end

    Phi1_fd(:,a) = (Xprop_p - Xprop_m) / (2*eps_fd);
end

err_phi1_fd = norm(Phi1_prop - Phi1_fd, 'fro') / max(norm(Phi1_prop, 'fro'), 1);
fprintf('  Phi1 FD relative error: %.3e  (eps=%.1e, dt=%.1e)\n', ...
    err_phi1_fd, eps_fd, dt_fd);
if err_phi1_fd < 1e-5
    fprintf('  Phi1 FD: PASS\n');
    pass_count = pass_count + 1;
else
    fprintf('  Phi1 FD: FAIL\n');
    fail_count = fail_count + 1;
end

% --- Phi2 via finite differences of Phi1 ---
% Phi2(i,a,b) = d Phi1(i,a) / d X_b(t0)
Phi2_fd = zeros(6,6,6);
for b = 1:6
    dX = zeros(6,1);
    dX(b) = eps_fd;

    % Forward: propagate Phi1 from X0+dX
    X_plus = X0 + dX;
    [~, P1d_p, ~, ~, ~] = cr3bp_stt_time_derivatives_mex(...
        X_plus, Phi1_0, Phi2_0, Phi3_0, Phi4_0, mu, Nfd);
    P1_prop_p = zeros(6,6);
    for k = 0:Nfd
        P1_prop_p = P1_prop_p + P1d_p(:,:,k+1) * dt_fd^k / factorial(k);
    end

    % Backward
    X_minus = X0 - dX;
    [~, P1d_m, ~, ~, ~] = cr3bp_stt_time_derivatives_mex(...
        X_minus, Phi1_0, Phi2_0, Phi3_0, Phi4_0, mu, Nfd);
    P1_prop_m = zeros(6,6);
    for k = 0:Nfd
        P1_prop_m = P1_prop_m + P1d_m(:,:,k+1) * dt_fd^k / factorial(k);
    end

    Phi2_fd(:,:,b) = (P1_prop_p - P1_prop_m) / (2*eps_fd);
end

err_phi2_fd = norm(Phi2_prop(:) - Phi2_fd(:)) / max(norm(Phi2_prop(:)), 1);
fprintf('  Phi2 FD relative error: %.3e  (eps=%.1e, dt=%.1e)\n', ...
    err_phi2_fd, eps_fd, dt_fd);
if err_phi2_fd < 1e-4
    fprintf('  Phi2 FD: PASS\n');
    pass_count = pass_count + 1;
else
    fprintf('  Phi2 FD: FAIL\n');
    fail_count = fail_count + 1;
end

fprintf('\n');

%% ======================================================================
%  Summary
%  ======================================================================
fprintf('=== SUMMARY: %d PASSED, %d FAILED ===\n', pass_count, fail_count);
if fail_count == 0
    fprintf('All tests passed!\n');
end

%% ======================================================================
%  Helper functions
%  ======================================================================

function Xdot = cr3bp_eom(~, X, mu)
    % CR3BP equations of motion (nondimensional rotating frame)
    x = X(1); y = X(2); z = X(3);
    vx = X(4); vy = X(5); vz = X(6);

    mu1 = 1 - mu;
    r1 = sqrt((x+mu)^2 + y^2 + z^2);
    r2 = sqrt((x-1+mu)^2 + y^2 + z^2);

    ax = 2*vy + x - mu1*(x+mu)/r1^3 - mu*(x-1+mu)/r2^3;
    ay = -2*vx + y - mu1*y/r1^3 - mu*y/r2^3;
    az = -mu1*z/r1^3 - mu*z/r2^3;

    Xdot = [vx; vy; vz; ax; ay; az];
end

function Ydot = cr3bp_stm_eom(~, Y, mu)
    % CR3BP state + STM variational equations
    X = Y(1:6);
    Phi = reshape(Y(7:42), 6, 6);

    Xdot = cr3bp_eom(0, X, mu);
    A = cr3bp_jacobian(X, mu);
    Phidot = A * Phi;

    Ydot = [Xdot; reshape(Phidot, 36, 1)];
end

function A = cr3bp_jacobian(X, mu)
    % CR3BP Jacobian (6x6) at state X
    x = X(1); y = X(2); z = X(3);
    mu1 = 1 - mu;

    r1sq = (x+mu)^2 + y^2 + z^2;
    r2sq = (x-1+mu)^2 + y^2 + z^2;
    r1_3 = r1sq^(-3/2);
    r2_3 = r2sq^(-3/2);
    r1_5 = r1sq^(-5/2);
    r2_5 = r2sq^(-5/2);

    q1 = x + mu;
    q2 = x - 1 + mu;

    Uxx = 1 - mu1*r1_3 + 3*mu1*q1^2*r1_5 - mu*r2_3 + 3*mu*q2^2*r2_5;
    Uyy = 1 - mu1*r1_3 + 3*mu1*y^2*r1_5 - mu*r2_3 + 3*mu*y^2*r2_5;
    Uzz = -mu1*r1_3 + 3*mu1*z^2*r1_5 - mu*r2_3 + 3*mu*z^2*r2_5;
    Uxy = 3*mu1*q1*y*r1_5 + 3*mu*q2*y*r2_5;
    Uxz = 3*mu1*q1*z*r1_5 + 3*mu*q2*z*r2_5;
    Uyz = 3*mu1*y*z*r1_5 + 3*mu*y*z*r2_5;

    A = zeros(6,6);
    A(1,4) = 1;
    A(2,5) = 1;
    A(3,6) = 1;
    A(4,1) = Uxx;  A(4,2) = Uxy;  A(4,3) = Uxz;  A(4,5) = 2;
    A(5,1) = Uxy;  A(5,2) = Uyy;  A(5,3) = Uyz;  A(5,4) = -2;
    A(6,1) = Uxz;  A(6,2) = Uyz;  A(6,3) = Uzz;
end
