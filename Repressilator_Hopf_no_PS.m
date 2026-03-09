% repressilator_sweep_noPS.m - NO PHASE SEPARATION (CANONICAL REPRESSILATOR)
%
% Modified to use ode23tb exclusively for all solver calls
% NO phase separation for any species (X, Y, Z all have no dense phases)
% Sweeping z* even though it doesn't affect dynamics (for comparison)
% Includes Hopf bifurcation diagram with z* as tuning parameter

clear; clc;
close all;

% =============================================================================
% PARAMETERS
% =============================================================================
alpha_x = 5; alpha_y = 5; alpha_z = 5;
tau_x = 1/0.02; tau_y = 1/0.02; tau_z = 1/0.02;
b_x = 1/tau_x;  b_y = 1/tau_y;  b_z = 1/tau_z;

n = 3;
K_xy = 20; K_yz = 20; K_zx = 20;

% Phase separation parameters are NOT USED
% z_star will be swept but has no effect

% Integration settings
t0 = 0; tf = 4500;
t_eval = linspace(t0, tf, 2250);
% State vector: [x_dilt; y_dilt; z_dilt]  (no dense phases)
y0 = [30; 38; 31];

% Sweep
z_star_values = 10:10:80;

% For verification plots
demo_vals = [10 20 40 60 80];
demo_store = struct();

% Outputs
amp_x = nan(size(z_star_values));
amp_y = nan(size(z_star_values));
amp_z = nan(size(z_star_values));
per_x = nan(size(z_star_values));
per_y = nan(size(z_star_values));
per_z = nan(size(z_star_values));

% BIFURCATION DIAGRAM OUTPUTS
bifurcation_x_max = nan(size(z_star_values));
bifurcation_x_min = nan(size(z_star_values));
bifurcation_y_max = nan(size(z_star_values));
bifurcation_y_min = nan(size(z_star_values));
bifurcation_z_max = nan(size(z_star_values));
bifurcation_z_min = nan(size(z_star_values));

% Measurement parameters - Consistent time windows
steady_state_duration = 1500;  % Analyze 1500 time units of steady state
steady_state_start = tf - steady_state_duration;  % = 3000
tail_window = steady_state_duration;  % Use same duration for peak analysis
max_intervals = 6;
amp_floor = 1.0;

% =============================================================================
% MAIN SWEEP LOOP - USING ODE23TB EXCLUSIVELY
% =============================================================================
fprintf('\n========================================\n');
fprintf('Running parameter sweep for z* from %d to %d (NO phase separation)\n', min(z_star_values), max(z_star_values));
fprintf('z* has NO EFFECT on dynamics (canonical repressilator)\n');
fprintf('========================================\n\n');

for i = 1:numel(z_star_values)
    z_star = z_star_values(i);  % Not used in dynamics, but kept for sweep

    % Create parameter structure for RHS (no phase separation parameters)
    params = struct(...
        'alpha_x', alpha_x, 'alpha_y', alpha_y, 'alpha_z', alpha_z, ...
        'b_x', b_x, 'b_y', b_y, 'b_z', b_z, ...
        'n', n, 'K_xy', K_xy, 'K_yz', K_yz, 'K_zx', K_zx);

    % RHS handle with parameters (no phase separation version)
    rhs = @(t,y) repressilator_rhs_noPS(t, y, params);

    % Solve with ode23tb
    [tSol, ySol, used_method, ok, msg] = solve_with_ode23tb(rhs, [t0 tf], y0, t_eval);

    fprintf('z_star=%5.1f  method=%s  ok=%d  t_end=%.2f  msg=%s\n', ...
        z_star, used_method, ok, tSol(end), msg);

    if ~ok
        continue;
    end

    % Store demos
    if any(demo_vals == z_star)
        key = sprintf('z%d', z_star);
        demo_store.(key) = struct('t', tSol, 'y', ySol, 'method', used_method);
    end

    % Extract dilute time series
    x_dilt = ySol(:,1);
    y_dilt = ySol(:,2);
    z_dilt = ySol(:,3);

    % Measure amplitude & period at the last observed peak
    [ax, px] = last_peak_amp_period(tSol, x_dilt, tail_window, max_intervals, amp_floor);
    [ay, py] = last_peak_amp_period(tSol, y_dilt, tail_window, max_intervals, amp_floor);
    [az, pz] = last_peak_amp_period(tSol, z_dilt, tail_window, max_intervals, amp_floor);

    amp_x(i) = ax; amp_y(i) = ay; amp_z(i) = az;
    per_x(i) = px; per_y(i) = py; per_z(i) = pz;
    
    % ===== HOPF BIFURCATION DATA COLLECTION =====
    % Use steady-state portion for bifurcation diagram
    idx_steady = tSol >= steady_state_start;
    if sum(idx_steady) > 100
        x_steady = x_dilt(idx_steady);
        y_steady = y_dilt(idx_steady);
        z_steady = z_dilt(idx_steady);
        
        % Store max and min for bifurcation diagram
        bifurcation_x_max(i) = max(x_steady);
        bifurcation_x_min(i) = min(x_steady);
        bifurcation_y_max(i) = max(y_steady);
        bifurcation_y_min(i) = min(y_steady);
        bifurcation_z_max(i) = max(z_steady);
        bifurcation_z_min(i) = min(z_steady);
    else
        % Not enough steady-state data
        bifurcation_x_max(i) = NaN;
        bifurcation_x_min(i) = NaN;
        bifurcation_y_max(i) = NaN;
        bifurcation_y_min(i) = NaN;
        bifurcation_z_max(i) = NaN;
        bifurcation_z_min(i) = NaN;
    end
end

fprintf('\n========================================\n');
fprintf('Sweep complete. Generating plots...\n');
fprintf('========================================\n\n');

% =============================================================================
% ORIGINAL PLOTS
% =============================================================================
figure('Color','w','Position',[100 100 1200 420]);
plot(z_star_values, amp_x, '-o', 'LineWidth',1.5); hold on;
plot(z_star_values, amp_y, '-o', 'LineWidth',1.5);
plot(z_star_values, amp_z, '-o', 'LineWidth',1.5);
grid on;
xlabel('z^*');
ylabel('Amplitude at last peak');
title('Amplitude vs z^* (No Phase Separation - Canonical Repressilator)');
legend('Amp(x_{dilt})','Amp(y_{dilt})','Amp(z_{dilt})','Location','best');

figure('Color','w','Position',[100 580 1200 420]);
plot(z_star_values, per_x, '-o', 'LineWidth',1.5); hold on;
plot(z_star_values, per_y, '-o', 'LineWidth',1.5);
plot(z_star_values, per_z, '-o', 'LineWidth',1.5);
grid on;
xlabel('z^*');
ylabel('Period');
title('Period vs z^* (No Phase Separation - Canonical Repressilator)');
legend('Per(x_{dilt})','Per(y_{dilt})','Per(z_{dilt})','Location','best');

% =============================================================================
% VERIFICATION TIME SERIES
% =============================================================================
for zs = demo_vals
    key = sprintf('z%d', zs);
    if ~isfield(demo_store, key)
        fprintf('Verification plot skipped for z_star=%d (integration failed)\n', zs);
        continue;
    end
    S = demo_store.(key);
    t = S.t; y = S.y;

    figure('Color','w','Position',[200 200 1200 420]);
    plot(t, y(:,1), 'r', 'LineWidth',1.2); hold on;
    plot(t, y(:,2), 'b', 'LineWidth',1.2);
    plot(t, y(:,3), 'g', 'LineWidth',1.2);
    grid on;
    xlabel('Time');
    ylabel('Copy Number');
    title(sprintf('Repressilator time series at z^* = %d (No Phase Separation)', zs));
    legend('x_{dilt}','y_{dilt}','z_{dilt}','Location','best');
    
    % Add vertical line indicating steady-state start
    xline(steady_state_start, 'k--', 'Steady State Start', 'LineWidth',1);
end

% =============================================================================
% HOPF BIFURCATION DIAGRAM
% =============================================================================
figure('Color','w','Position',[300 300 1400 1000]);

% X species bifurcation (no phase separation)
subplot(3,1,1);
hold on;
% Plot max and min as points
plot(z_star_values, bifurcation_x_max, 'r.', 'MarkerSize', 12);
plot(z_star_values, bifurcation_x_min, 'r.', 'MarkerSize', 12);
% Connect stable branches with lines
for i = 1:length(z_star_values)-1
    if ~isnan(bifurcation_x_max(i)) && ~isnan(bifurcation_x_max(i+1))
        plot([z_star_values(i), z_star_values(i+1)], ...
             [bifurcation_x_max(i), bifurcation_x_max(i+1)], 'r-', 'LineWidth', 1);
        plot([z_star_values(i), z_star_values(i+1)], ...
             [bifurcation_x_min(i), bifurcation_x_min(i+1)], 'r-', 'LineWidth', 1);
    end
end
% Fill the oscillation envelope
for i = 1:length(z_star_values)
    if ~isnan(bifurcation_x_max(i)) && ~isnan(bifurcation_x_min(i))
        x_patch = [z_star_values(i)-2, z_star_values(i)+2, z_star_values(i)+2, z_star_values(i)-2];
        y_patch = [bifurcation_x_min(i), bifurcation_x_min(i), bifurcation_x_max(i), bifurcation_x_max(i)];
        patch(x_patch, y_patch, 'r', 'FaceAlpha', 0.2, 'EdgeColor', 'none');
    end
end
grid on; box on;
xlabel('z^* (No Effect)', 'FontSize', 12, 'FontWeight', 'bold');
ylabel('x_{dilt} Copy Number', 'FontSize', 12, 'FontWeight', 'bold');
title('Hopf Bifurcation: x_{dilt} (No Phase Separation)', 'FontSize', 14, 'FontWeight', 'bold');
% Mark that z* has no effect
text(0.5, 0.9, 'z^* has NO EFFECT on dynamics', 'Units', 'normalized', ...
     'FontSize', 11, 'BackgroundColor', 'yellow', 'EdgeColor', 'k');

% Y species bifurcation (no phase separation)
subplot(3,1,2);
hold on;
plot(z_star_values, bifurcation_y_max, 'b.', 'MarkerSize', 12);
plot(z_star_values, bifurcation_y_min, 'b.', 'MarkerSize', 12);
for i = 1:length(z_star_values)-1
    if ~isnan(bifurcation_y_max(i)) && ~isnan(bifurcation_y_max(i+1))
        plot([z_star_values(i), z_star_values(i+1)], ...
             [bifurcation_y_max(i), bifurcation_y_max(i+1)], 'b-', 'LineWidth', 1);
        plot([z_star_values(i), z_star_values(i+1)], ...
             [bifurcation_y_min(i), bifurcation_y_min(i+1)], 'b-', 'LineWidth', 1);
    end
end
for i = 1:length(z_star_values)
    if ~isnan(bifurcation_y_max(i)) && ~isnan(bifurcation_y_min(i))
        x_patch = [z_star_values(i)-2, z_star_values(i)+2, z_star_values(i)+2, z_star_values(i)-2];
        y_patch = [bifurcation_y_min(i), bifurcation_y_min(i), bifurcation_y_max(i), bifurcation_y_max(i)];
        patch(x_patch, y_patch, 'b', 'FaceAlpha', 0.2, 'EdgeColor', 'none');
    end
end
grid on; box on;
xlabel('z^* (No Effect)', 'FontSize', 12, 'FontWeight', 'bold');
ylabel('y_{dilt} Copy Number', 'FontSize', 12, 'FontWeight', 'bold');
title('Hopf Bifurcation: y_{dilt} (No Phase Separation)', 'FontSize', 14, 'FontWeight', 'bold');
text(0.5, 0.9, 'z^* has NO EFFECT on dynamics', 'Units', 'normalized', ...
     'FontSize', 11, 'BackgroundColor', 'yellow', 'EdgeColor', 'k');

% Z species bifurcation (no phase separation)
subplot(3,1,3);
hold on;
plot(z_star_values, bifurcation_z_max, 'g.', 'MarkerSize', 12);
plot(z_star_values, bifurcation_z_min, 'g.', 'MarkerSize', 12);
for i = 1:length(z_star_values)-1
    if ~isnan(bifurcation_z_max(i)) && ~isnan(bifurcation_z_max(i+1))
        plot([z_star_values(i), z_star_values(i+1)], ...
             [bifurcation_z_max(i), bifurcation_z_max(i+1)], 'g-', 'LineWidth', 1);
        plot([z_star_values(i), z_star_values(i+1)], ...
             [bifurcation_z_min(i), bifurcation_z_min(i+1)], 'g-', 'LineWidth', 1);
    end
end
for i = 1:length(z_star_values)
    if ~isnan(bifurcation_z_max(i)) && ~isnan(bifurcation_z_min(i))
        x_patch = [z_star_values(i)-2, z_star_values(i)+2, z_star_values(i)+2, z_star_values(i)-2];
        y_patch = [bifurcation_z_min(i), bifurcation_z_min(i), bifurcation_z_max(i), bifurcation_z_max(i)];
        patch(x_patch, y_patch, 'g', 'FaceAlpha', 0.2, 'EdgeColor', 'none');
    end
end
grid on; box on;
xlabel('z^* (No Effect)', 'FontSize', 12, 'FontWeight', 'bold');
ylabel('z_{dilt} Copy Number', 'FontSize', 12, 'FontWeight', 'bold');
title('Hopf Bifurcation: z_{dilt} (No Phase Separation)', 'FontSize', 14, 'FontWeight', 'bold');
text(0.5, 0.9, 'z^* has NO EFFECT on dynamics', 'Units', 'normalized', ...
     'FontSize', 11, 'BackgroundColor', 'yellow', 'EdgeColor', 'k');

sgtitle('Hopf Bifurcation Diagram with z^* as Tuning Parameter (NO Phase Separation - Canonical Repressilator)', ...
    'FontSize', 16, 'FontWeight', 'bold');

% =============================================================================
% COMBINED BIFURCATION DIAGRAM (ALL SPECIES)
% =============================================================================
figure('Color','w','Position',[350 350 1200 600]);

hold on;
% X species
plot(z_star_values, bifurcation_x_max, 'r.', 'MarkerSize', 10);
plot(z_star_values, bifurcation_x_min, 'r.', 'MarkerSize', 10);
% Y species
plot(z_star_values, bifurcation_y_max, 'b.', 'MarkerSize', 10);
plot(z_star_values, bifurcation_y_min, 'b.', 'MarkerSize', 10);
% Z species
plot(z_star_values, bifurcation_z_max, 'g.', 'MarkerSize', 10);
plot(z_star_values, bifurcation_z_min, 'g.', 'MarkerSize', 10);

% Connect branches
for i = 1:length(z_star_values)-1
    if ~isnan(bifurcation_x_max(i)) && ~isnan(bifurcation_x_max(i+1))
        plot([z_star_values(i), z_star_values(i+1)], ...
             [bifurcation_x_max(i), bifurcation_x_max(i+1)], 'r-', 'LineWidth', 0.5);
        plot([z_star_values(i), z_star_values(i+1)], ...
             [bifurcation_x_min(i), bifurcation_x_min(i+1)], 'r-', 'LineWidth', 0.5);
    end
    if ~isnan(bifurcation_y_max(i)) && ~isnan(bifurcation_y_max(i+1))
        plot([z_star_values(i), z_star_values(i+1)], ...
             [bifurcation_y_max(i), bifurcation_y_max(i+1)], 'b-', 'LineWidth', 0.5);
        plot([z_star_values(i), z_star_values(i+1)], ...
             [bifurcation_y_min(i), bifurcation_y_min(i+1)], 'b-', 'LineWidth', 0.5);
    end
    if ~isnan(bifurcation_z_max(i)) && ~isnan(bifurcation_z_max(i+1))
        plot([z_star_values(i), z_star_values(i+1)], ...
             [bifurcation_z_max(i), bifurcation_z_max(i+1)], 'g-', 'LineWidth', 0.5);
        plot([z_star_values(i), z_star_values(i+1)], ...
             [bifurcation_z_min(i), bifurcation_z_min(i+1)], 'g-', 'LineWidth', 0.5);
    end
end

grid on; box on;
xlabel('z^* (No Effect - Swept for Comparison)', 'FontSize', 14, 'FontWeight', 'bold');
ylabel('Copy Number', 'FontSize', 14, 'FontWeight', 'bold');
title('Combined Bifurcation Diagram (NO Phase Separation - Canonical Repressilator)', 'FontSize', 16, 'FontWeight', 'bold');
legend({'x_{dilt}', 'y_{dilt}', 'z_{dilt}'}, ...
    'Location', 'best', 'FontSize', 12);

% Add text annotation explaining that z* has no effect
xlim([min(z_star_values)-2, max(z_star_values)+2]);
ylim_orig = ylim;
text(min(z_star_values)+2, ylim_orig(2)*0.9, ...
    'z^* has NO EFFECT on dynamics (canonical repressilator)', ...
    'FontSize', 12, 'BackgroundColor', 'yellow', 'EdgeColor', 'k');

% =============================================================================
% DATA EXPORT BLOCK - EXPORTS ALL PLOTTED DATA TO FILES
% =============================================================================
fprintf('\n========================================\n');
fprintf('Exporting data to files...\n');
fprintf('========================================\n\n');

% Create data directory
export_dir = 'exported_data';
if ~exist(export_dir, 'dir')
    mkdir(export_dir);
end

% 1. Export amplitude vs z_star data
fid = fopen(fullfile(export_dir, 'amplitude_vs_zstar_noPS.dat'), 'w');
fprintf(fid, 'z_star\tamp_x\tamp_y\tamp_z\n');
for i = 1:length(z_star_values)
    fprintf(fid, '%.1f\t%.6f\t%.6f\t%.6f\n', z_star_values(i), amp_x(i), amp_y(i), amp_z(i));
end
fclose(fid);
fprintf('Exported: amplitude_vs_zstar_noPS.dat\n');

% 2. Export period vs z_star data
fid = fopen(fullfile(export_dir, 'period_vs_zstar_noPS.dat'), 'w');
fprintf(fid, 'z_star\tper_x\tper_y\tper_z\n');
for i = 1:length(z_star_values)
    fprintf(fid, '%.1f\t%.6f\t%.6f\t%.6f\n', z_star_values(i), per_x(i), per_y(i), per_z(i));
end
fclose(fid);
fprintf('Exported: period_vs_zstar_noPS.dat\n');

% 3. Export bifurcation data - separate files for each species
% X species
fid = fopen(fullfile(export_dir, 'x_branch_noPS.dat'), 'w');
fprintf(fid, 'z_star\tx_max\tx_min\n');
for i = 1:length(z_star_values)
    if ~isnan(bifurcation_x_max(i))
        fprintf(fid, '%.1f\t%.6f\t%.6f\n', z_star_values(i), bifurcation_x_max(i), bifurcation_x_min(i));
    end
end
fclose(fid);
fprintf('Exported: x_branch_noPS.dat\n');

% Y species
fid = fopen(fullfile(export_dir, 'y_branch_noPS.dat'), 'w');
fprintf(fid, 'z_star\ty_max\ty_min\n');
for i = 1:length(z_star_values)
    if ~isnan(bifurcation_y_max(i))
        fprintf(fid, '%.1f\t%.6f\t%.6f\n', z_star_values(i), bifurcation_y_max(i), bifurcation_y_min(i));
    end
end
fclose(fid);
fprintf('Exported: y_branch_noPS.dat\n');

% Z species
fid = fopen(fullfile(export_dir, 'z_branch_noPS.dat'), 'w');
fprintf(fid, 'z_star\tz_max\tz_min\n');
for i = 1:length(z_star_values)
    if ~isnan(bifurcation_z_max(i))
        fprintf(fid, '%.1f\t%.6f\t%.6f\n', z_star_values(i), bifurcation_z_max(i), bifurcation_z_min(i));
    end
end
fclose(fid);
fprintf('Exported: z_branch_noPS.dat\n');

% 4. Export time series data for demo values
for zs = demo_vals
    key = sprintf('z%d', zs);
    if isfield(demo_store, key)
        S = demo_store.(key);
        t = S.t;
        y = S.y;
        
        filename = fullfile(export_dir, sprintf('timeseries_z%d_noPS.dat', zs));
        fid = fopen(filename, 'w');
        fprintf(fid, 'time\tx_dilt\ty_dilt\tz_dilt\n');
        
        % Decimate to ~1000 points for manageable file size
        step = max(1, floor(length(t)/1000));
        for i = 1:step:length(t)
            fprintf(fid, '%.6f\t%.6f\t%.6f\t%.6f\n', ...
                t(i), y(i,1), y(i,2), y(i,3));
        end
        fclose(fid);
        fprintf('Exported: timeseries_z%d_noPS.dat\n', zs);
    end
end


% =============================================================================
% ========================= LOCAL FUNCTIONS ===================================
% =============================================================================

function dydt = repressilator_rhs_noPS(~, y, p)
    % NO PHASE SEPARATION VERSION: Canonical repressilator
    % X, Y, Z have no dense phases
    % State vector: [x_dilt; y_dilt; z_dilt]
    
    % State variables - ensure non-negative
    x_dilt = max(y(1), 0);
    y_dilt = max(y(2), 0);
    z_dilt = max(y(3), 0);

    % Production terms (repression) - canonical repressilator
    prod_x = p.alpha_x / (1 + (z_dilt / p.K_zx)^p.n);
    prod_y = p.alpha_y / (1 + (x_dilt / p.K_xy)^p.n);
    prod_z = p.alpha_z / (1 + (y_dilt / p.K_yz)^p.n);

    % Simple degradation only (no phase separation terms)
    dxdilt_dt = prod_x - p.b_x * x_dilt;
    dydilt_dt = prod_y - p.b_y * y_dilt;
    dzdilt_dt = prod_z - p.b_z * z_dilt;

    % Return derivatives in same order as state vector
    dydt = [dxdilt_dt; dydilt_dt; dzdilt_dt];
end

function [tSol, ySol, used_method, ok, msg] = solve_with_ode23tb(rhs, tspan2, y0, t_eval)
    % Modified to use ONLY ode23tb with progressively stricter tolerances
    
    ok = false;
    msg = "failed";
    
    % Try ode23tb with different tolerance settings
    tolerance_attempts = {
        'default',      odeset();                          % Default tolerances
        'moderate',     odeset('RelTol', 1e-4, 'AbsTol', 1e-7);  % Moderate tolerances
        'strict',       odeset('RelTol', 1e-6, 'AbsTol', 1e-9);  % Strict tolerances
        'very_strict',  odeset('RelTol', 1e-8, 'AbsTol', 1e-11); % Very strict tolerances
    };
    
    last_t = []; last_y = [];
    
    for k = 1:size(tolerance_attempts, 1)
        tol_name = tolerance_attempts{k,1};
        opts = tolerance_attempts{k,2};
        used_method = sprintf('ode23tb (%s tolerances)', tol_name);
        
        try
            [t, y] = ode23tb(rhs, t_eval, y0, opts);
            
            last_t = t; last_y = y;
            
            if ~isempty(t) && t(end) >= 0.999 * tspan2(2)
                ok = true;
                msg = sprintf("success with %s tolerances", tol_name);
                tSol = t; ySol = y;
                return;
            else
                msg = sprintf("ended early with %s tolerances", tol_name);
            end
            
        catch ME
            msg = sprintf("ode23tb (%s) failed: %s", tol_name, ME.message);
            last_t = []; last_y = [];
        end
    end
    
    % If all tolerance settings failed, try one more time with outputfcn to debug
    try
        opts = odeset('RelTol', 1e-4, 'AbsTol', 1e-7, 'OutputFcn', @odeoutput);
        [t, y] = ode23tb(rhs, t_eval, y0, opts);
        last_t = t; last_y = y;
        used_method = 'ode23tb (final attempt with output)';
        msg = "completed but may have issues";
    catch ME
        msg = sprintf("All ode23tb attempts failed: %s", ME.message);
    end
    
    % Return whatever we have (even if incomplete)
    if isempty(last_t)
        tSol = t_eval(:);
        ySol = nan(numel(t_eval), numel(y0));
    else
        tSol = last_t;
        ySol = last_y;
    end
end

function status = odeoutput(t, y, flag)
    % Simple output function to monitor integration progress
    persistent t_start
    
    status = 0;  % Continue integration
    
    if isempty(flag)
        % Integration step
        if t > t_start + 100
            fprintf('  ode23tb: t = %.1f\n', t);
            t_start = t;
        end
    elseif strcmp(flag, 'init')
        % Start of integration
        t_start = t;
        fprintf('  Starting ode23tb integration...\n');
    elseif strcmp(flag, 'done')
        % End of integration
        fprintf('  Finished ode23tb integration at t = %.1f\n', t);
    end
end

function [amp, period] = last_peak_amp_period(t, s, tail_window, max_intervals, amp_floor)
    s = s(:); t = t(:);

    s_scale = prctile(s,95) - prctile(s,5);
    if s_scale <= 0
        amp = 0.0; period = NaN; return;
    end

    t_end = t(end);
    t0 = max(t(1), t_end - tail_window);
    idx_tail = (t >= t0);
    tt = t(idx_tail);
    ss = s(idx_tail);

    per_est = estimate_main_period_fft(tt, ss, 20.0, 4000.0);
    dt = median(diff(t));

    if isnan(per_est)
        distPts = max(1, floor(50.0 / dt));
    else
        distPts = max(1, floor(0.45 * per_est / dt));
    end

    prom = max(1e-6, 0.15 * s_scale);

    % Use MATLAB findpeaks if available, else fallback
    if exist('findpeaks','file') == 2
        [~, pIdx] = findpeaks(s, 'MinPeakProminence', prom, 'MinPeakDistance', distPts);
        [~, qIdx] = findpeaks(-s, 'MinPeakProminence', prom, 'MinPeakDistance', distPts);
    else
        pIdx = findpeaks_simple(s, prom, distPts);
        qIdx = findpeaks_simple(-s, prom, distPts);
    end

    if isempty(pIdx) || isempty(qIdx)
        amp = 0.0; period = NaN; return;
    end

    p_last = pIdx(end);
    q_before = qIdx(qIdx < p_last);
    if isempty(q_before)
        amp = 0.0; period = NaN; return;
    end
    q_last = q_before(end);

    amp = 0.5 * (s(p_last) - s(q_last));
    if amp < amp_floor
        amp = 0.0; period = NaN; return;
    end

    if numel(pIdx) < 2
        period = NaN; return;
    end

    peak_times = t(pIdx);
    intervals = diff(peak_times);
    if isempty(intervals)
        period = NaN; return;
    end
    k = min(max_intervals, numel(intervals));
    period = median(intervals(end-k+1:end));
end

function per = estimate_main_period_fft(t, s, min_period, max_period)
    t = t(:); s = s(:);
    if numel(s) < 2048
        per = NaN; return;
    end
    dt = median(diff(t));
    x = s - mean(s);
    N = numel(x);

    Y = fft(x);
    nHalf = floor(N/2);
    spec = abs(Y(1:nHalf+1)).^2;
    freqs = (0:nHalf)' / (N*dt);

    fmin = 1.0 / max_period;
    fmax = 1.0 / min_period;
    mask = (freqs >= fmin) & (freqs <= fmax);

    if sum(mask) < 10
        per = NaN; return;
    end

    spec_m = spec(mask);
    freqs_m = freqs(mask);
    [~, imax] = max(spec_m);
    f0 = freqs_m(imax);
    if f0 <= 0
        per = NaN;
    else
        per = 1.0 / f0;
    end
end

function idx = findpeaks_simple(x, prom, distPts)
    x = x(:);
    dx = diff(x);
    sgn = sign(dx);
    sgn(sgn==0) = 1;
    turn = diff(sgn);
    cand = find(turn < 0) + 1;  % local maxima indices

    % crude prominence filter: keep peaks above median + prom
    thr = median(x) + prom;
    cand = cand(x(cand) >= thr);

    if isempty(cand)
        idx = [];
        return;
    end

    % enforce minimum distance
    idx = cand(1);
    for k = 2:numel(cand)
        if cand(k) - idx(end) >= distPts
            idx(end+1,1) = cand(k);
        end
    end
end