% repressilator_phase_sep_sweep_YZ.m - YZ PHASE SEPARATION ONLY
%
% Modified to use robust oscillation analysis
% X does NOT participate in phase separation (no dense phase)
% Y and Z participate in phase separation
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

v = 1.0e-25;
volume_factor = 1e5;
Vtot = volume_factor * v;

% Phase separation thresholds (only for Y and Z)
y_star = 80;
% z_star will be swept

thermal_energy = 1.38e-23 * 305;
tau_d = 0.1 * tau_z;
D = Vtot^(2/3) / (6 * tau_d);

% Chemical potentials (only for Y and Z)
phi_star_y = y_star / volume_factor;
mu_y = -thermal_energy * (log(phi_star_y) - phi_star_y);
% mu_z will be computed for each z_star value

% Integration settings
t0 = 0; tf = 4000;
t_eval = linspace(t0, tf, 2250);
% State vector: [x_dilt; y_dilt; y_dens; z_dilt; z_dens]
y0 = [30; 38; 0.0; 31; 0.0];

% Sweep
% z_star_values = 10:10:80;
z_star_values = [10:5:30, 31:1:40, 45:5:80];

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
osc_x = false(size(z_star_values));
osc_y = false(size(z_star_values));
osc_z = false(size(z_star_values));

% BIFURCATION DIAGRAM OUTPUTS
bifurcation_x_max = nan(size(z_star_values));
bifurcation_x_min = nan(size(z_star_values));
bifurcation_y_max = nan(size(z_star_values));
bifurcation_y_min = nan(size(z_star_values));
bifurcation_z_max = nan(size(z_star_values));
bifurcation_z_min = nan(size(z_star_values));

% Measurement parameters - Consistent time windows
steady_state_duration = 1000;  % Analyze 1500 time units of steady state
steady_state_start = tf - steady_state_duration;  % = 3000

% =============================================================================
% MAIN SWEEP LOOP - USING ODE23TB EXCLUSIVELY
% =============================================================================
fprintf('\n========================================\n');
fprintf('Running parameter sweep for z* from %d to %d (YZ phase separation only)\n', min(z_star_values), max(z_star_values));
fprintf('========================================\n\n');

for i = 1:numel(z_star_values)
    z_star = z_star_values(i);

    % z-dependent chemical potential
    phi_star_z = z_star / volume_factor;
    mu_z = -thermal_energy * (log(phi_star_z) - phi_star_z);

    % Create parameter structure for RHS
    params = struct(...
        'alpha_x', alpha_x, 'alpha_y', alpha_y, 'alpha_z', alpha_z, ...
        'b_x', b_x, 'b_y', b_y, 'b_z', b_z, ...
        'n', n, 'K_xy', K_xy, 'K_yz', K_yz, 'K_zx', K_zx, ...
        'v', v, 'Vtot', Vtot, ...
        'y_star', y_star, 'z_star', z_star, ...
        'thermal_energy', thermal_energy, 'D', D, ...
        'volume_factor', volume_factor, ...
        'mu_y', mu_y, 'mu_z', mu_z);

    % RHS handle with parameters (using YZ-only version)
    rhs = @(t,y) repressilator_rhs_YZ_only(t, y, params);

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
    z_dilt = ySol(:,4);
    
    % ===== HOPF BIFURCATION DATA COLLECTION =====
    % Use steady-state portion for bifurcation diagram
    idx_steady = tSol >= steady_state_start;
    if sum(idx_steady) > 100
        x_steady = x_dilt(idx_steady);
        y_steady = y_dilt(idx_steady);
        z_steady = z_dilt(idx_steady);
        t_steady = tSol(idx_steady);
        
        % Robust oscillation analysis for each species
        min_cycles = 3;  % Require at least 3 complete cycles
        
        [amp_x(i), per_x(i), osc_x(i)] = robust_oscillation_analysis(...
            t_steady, x_steady, min_cycles);
        [amp_y(i), per_y(i), osc_y(i)] = robust_oscillation_analysis(...
            t_steady, y_steady, min_cycles);
        [amp_z(i), per_z(i), osc_z(i)] = robust_oscillation_analysis(...
            t_steady, z_steady, min_cycles);
        
        % Store max and min for bifurcation diagram
        if osc_x(i)
            bifurcation_x_max(i) = max(x_steady);
            bifurcation_x_min(i) = min(x_steady);
        else
            % For non-oscillatory, store mean as both max and min
            bifurcation_x_max(i) = mean(x_steady);
            bifurcation_x_min(i) = mean(x_steady);
        end
        
        if osc_y(i)
            bifurcation_y_max(i) = max(y_steady);
            bifurcation_y_min(i) = min(y_steady);
        else
            bifurcation_y_max(i) = mean(y_steady);
            bifurcation_y_min(i) = mean(y_steady);
        end
        
        if osc_z(i)
            bifurcation_z_max(i) = max(z_steady);
            bifurcation_z_min(i) = min(z_steady);
        else
            bifurcation_z_max(i) = mean(z_steady);
            bifurcation_z_min(i) = mean(z_steady);
        end
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
plot(z_star_values, amp_x, '-o', 'LineWidth',1.5, 'MarkerFaceColor','r'); hold on;
plot(z_star_values, amp_y, '-o', 'LineWidth',1.5, 'MarkerFaceColor','b');
plot(z_star_values, amp_z, '-o', 'LineWidth',1.5, 'MarkerFaceColor','g');
grid on;
xlabel('z^*');
ylabel('Amplitude');
title('Amplitude vs z^* (YZ Phase Separation Only) - Robust Analysis');
legend('Amp(x_{dilt})','Amp(y_{dilt})','Amp(z_{dilt})','Location','best');

% Mark non-oscillatory regions
for i = 1:numel(z_star_values)
    if ~osc_x(i)
        plot(z_star_values(i), amp_x(i), 'rx', 'MarkerSize', 12, 'LineWidth', 2);
    end
    if ~osc_y(i)
        plot(z_star_values(i), amp_y(i), 'bx', 'MarkerSize', 12, 'LineWidth', 2);
    end
    if ~osc_z(i)
        plot(z_star_values(i), amp_z(i), 'gx', 'MarkerSize', 12, 'LineWidth', 2);
    end
end

figure('Color','w','Position',[100 580 1200 420]);
plot(z_star_values, per_x, '-o', 'LineWidth',1.5, 'MarkerFaceColor','r'); hold on;
plot(z_star_values, per_y, '-o', 'LineWidth',1.5, 'MarkerFaceColor','b');
plot(z_star_values, per_z, '-o', 'LineWidth',1.5, 'MarkerFaceColor','g');
grid on;
xlabel('z^*');
ylabel('Period');
title('Period vs z^* (YZ Phase Separation Only) - Robust Analysis');
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
    plot(t, y(:,4), 'g', 'LineWidth',1.2);
    grid on;
    xlabel('Time');
    ylabel('Copy Number');
    title(sprintf('Repressilator time series at z^* = %d (YZ Phase Separation Only)', zs));
    legend('x_{dilt}','y_{dilt}','z_{dilt}','Location','best');
    
    % Add vertical line indicating steady-state start
    xline(steady_state_start, 'k--', 'Steady State Start', 'LineWidth',1);
    
    % Mark the analyzed region
    xregion(steady_state_start, tf, 'FaceColor', [0.9 0.9 0.9], 'FaceAlpha', 0.3);
end

% =============================================================================
% HOPF BIFURCATION DIAGRAM
% =============================================================================
figure('Color','w','Position',[300 300 1400 1000]);

% X species bifurcation (no phase separation)
subplot(3,1,1);
hold on;
% Plot max and min as points
for i = 1:length(z_star_values)
    if osc_x(i)
        % Oscillatory - plot both max and min
        plot(z_star_values(i), bifurcation_x_max(i), 'r.', 'MarkerSize', 12);
        plot(z_star_values(i), bifurcation_x_min(i), 'r.', 'MarkerSize', 12);
    else
        % Non-oscillatory - plot single point
        plot(z_star_values(i), bifurcation_x_max(i), 'ro', 'MarkerSize', 8, 'MarkerFaceColor','r');
    end
end
% Connect stable branches with lines
for i = 1:length(z_star_values)-1
    if ~isnan(bifurcation_x_max(i)) && ~isnan(bifurcation_x_max(i+1))
        plot([z_star_values(i), z_star_values(i+1)], ...
             [bifurcation_x_max(i), bifurcation_x_max(i+1)], 'r-', 'LineWidth', 1);
        plot([z_star_values(i), z_star_values(i+1)], ...
             [bifurcation_x_min(i), bifurcation_x_min(i+1)], 'r-', 'LineWidth', 1);
    end
end
% Fill the oscillation envelope where oscillatory
for i = 1:length(z_star_values)
    if osc_x(i) && ~isnan(bifurcation_x_max(i)) && ~isnan(bifurcation_x_min(i))
        x_patch = [z_star_values(i)-2, z_star_values(i)+2, z_star_values(i)+2, z_star_values(i)-2];
        y_patch = [bifurcation_x_min(i), bifurcation_x_min(i), bifurcation_x_max(i), bifurcation_x_max(i)];
        patch(x_patch, y_patch, 'r', 'FaceAlpha', 0.2, 'EdgeColor', 'none');
    end
end
grid on; box on;
xlabel('z^*', 'FontSize', 12, 'FontWeight', 'bold');
ylabel('x_{dilt} Copy Number', 'FontSize', 12, 'FontWeight', 'bold');
title('Hopf Bifurcation: x_{dilt} (No Phase Separation)', 'FontSize', 14, 'FontWeight', 'bold');
% Mark Hopf bifurcation point (where oscillations begin)
first_osc = find(osc_x, 1);
if ~isempty(first_osc) && first_osc > 1
    xline(z_star_values(first_osc), 'b--', 'LineWidth', 2, 'Label', 'Hopf Point');
end

% Y species bifurcation (with phase separation)
subplot(3,1,2);
hold on;
for i = 1:length(z_star_values)
    if osc_y(i)
        plot(z_star_values(i), bifurcation_y_max(i), 'b.', 'MarkerSize', 12);
        plot(z_star_values(i), bifurcation_y_min(i), 'b.', 'MarkerSize', 12);
    else
        plot(z_star_values(i), bifurcation_y_max(i), 'bo', 'MarkerSize', 8, 'MarkerFaceColor','b');
    end
end
for i = 1:length(z_star_values)-1
    if ~isnan(bifurcation_y_max(i)) && ~isnan(bifurcation_y_max(i+1))
        plot([z_star_values(i), z_star_values(i+1)], ...
             [bifurcation_y_max(i), bifurcation_y_max(i+1)], 'b-', 'LineWidth', 1);
        plot([z_star_values(i), z_star_values(i+1)], ...
             [bifurcation_y_min(i), bifurcation_y_min(i+1)], 'b-', 'LineWidth', 1);
    end
end
for i = 1:length(z_star_values)
    if osc_y(i) && ~isnan(bifurcation_y_max(i)) && ~isnan(bifurcation_y_min(i))
        x_patch = [z_star_values(i)-2, z_star_values(i)+2, z_star_values(i)+2, z_star_values(i)-2];
        y_patch = [bifurcation_y_min(i), bifurcation_y_min(i), bifurcation_y_max(i), bifurcation_y_max(i)];
        patch(x_patch, y_patch, 'b', 'FaceAlpha', 0.2, 'EdgeColor', 'none');
    end
end
grid on; box on;
xlabel('z^*', 'FontSize', 12, 'FontWeight', 'bold');
ylabel('y_{dilt} Copy Number', 'FontSize', 12, 'FontWeight', 'bold');
title('Hopf Bifurcation: y_{dilt} (With Phase Separation)', 'FontSize', 14, 'FontWeight', 'bold');
first_osc = find(osc_y, 1);
if ~isempty(first_osc) && first_osc > 1
    xline(z_star_values(first_osc), 'b--', 'LineWidth', 2, 'Label', 'Hopf Point');
end

% Z species bifurcation (with phase separation)
subplot(3,1,3);
hold on;
for i = 1:length(z_star_values)
    if osc_z(i)
        plot(z_star_values(i), bifurcation_z_max(i), 'g.', 'MarkerSize', 12);
        plot(z_star_values(i), bifurcation_z_min(i), 'g.', 'MarkerSize', 12);
    else
        plot(z_star_values(i), bifurcation_z_max(i), 'go', 'MarkerSize', 8, 'MarkerFaceColor','g');
    end
end
for i = 1:length(z_star_values)-1
    if ~isnan(bifurcation_z_max(i)) && ~isnan(bifurcation_z_max(i+1))
        plot([z_star_values(i), z_star_values(i+1)], ...
             [bifurcation_z_max(i), bifurcation_z_max(i+1)], 'g-', 'LineWidth', 1);
        plot([z_star_values(i), z_star_values(i+1)], ...
             [bifurcation_z_min(i), bifurcation_z_min(i+1)], 'g-', 'LineWidth', 1);
    end
end
for i = 1:length(z_star_values)
    if osc_z(i) && ~isnan(bifurcation_z_max(i)) && ~isnan(bifurcation_z_min(i))
        x_patch = [z_star_values(i)-2, z_star_values(i)+2, z_star_values(i)+2, z_star_values(i)-2];
        y_patch = [bifurcation_z_min(i), bifurcation_z_min(i), bifurcation_z_max(i), bifurcation_z_max(i)];
        patch(x_patch, y_patch, 'g', 'FaceAlpha', 0.2, 'EdgeColor', 'none');
    end
end
grid on; box on;
xlabel('z^*', 'FontSize', 12, 'FontWeight', 'bold');
ylabel('z_{dilt} Copy Number', 'FontSize', 12, 'FontWeight', 'bold');
title('Hopf Bifurcation: z_{dilt} (With Phase Separation)', 'FontSize', 14, 'FontWeight', 'bold');
first_osc = find(osc_z, 1);
if ~isempty(first_osc) && first_osc > 1
    xline(z_star_values(first_osc), 'b--', 'LineWidth', 2, 'Label', 'Hopf Point');
end

sgtitle('Hopf Bifurcation Diagram with z^* as Tuning Parameter (YZ Phase Separation Only)', ...
    'FontSize', 16, 'FontWeight', 'bold');

% =============================================================================
% COMBINED BIFURCATION DIAGRAM (ALL SPECIES)
% =============================================================================
figure('Color','w','Position',[350 350 1200 600]);

hold on;
% X species
for i = 1:length(z_star_values)
    if osc_x(i)
        plot(z_star_values(i), bifurcation_x_max(i), 'r.', 'MarkerSize', 10);
        plot(z_star_values(i), bifurcation_x_min(i), 'r.', 'MarkerSize', 10);
    else
        plot(z_star_values(i), bifurcation_x_max(i), 'ro', 'MarkerSize', 6, 'MarkerFaceColor','r');
    end
end
% Y species
for i = 1:length(z_star_values)
    if osc_y(i)
        plot(z_star_values(i), bifurcation_y_max(i), 'b.', 'MarkerSize', 10);
        plot(z_star_values(i), bifurcation_y_min(i), 'b.', 'MarkerSize', 10);
    else
        plot(z_star_values(i), bifurcation_y_max(i), 'bo', 'MarkerSize', 6, 'MarkerFaceColor','b');
    end
end
% Z species
for i = 1:length(z_star_values)
    if osc_z(i)
        plot(z_star_values(i), bifurcation_z_max(i), 'g.', 'MarkerSize', 10);
        plot(z_star_values(i), bifurcation_z_min(i), 'g.', 'MarkerSize', 10);
    else
        plot(z_star_values(i), bifurcation_z_max(i), 'go', 'MarkerSize', 6, 'MarkerFaceColor','g');
    end
end

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
xlabel('z^* (Phase Separation Threshold for Y and Z)', 'FontSize', 14, 'FontWeight', 'bold');
ylabel('Copy Number', 'FontSize', 14, 'FontWeight', 'bold');
title('Combined Hopf Bifurcation Diagram (YZ Phase Separation Only)', 'FontSize', 16, 'FontWeight', 'bold');
legend({'x_{dilt} (no PS)', '', 'y_{dilt} (with PS)', '', 'z_{dilt} (with PS)', ''}, ...
    'Location', 'best', 'FontSize', 12);

% Mark Hopf bifurcation region
first_osc_all = min([find(osc_x, 1); find(osc_y, 1); find(osc_z, 1)]);
if ~isempty(first_osc_all) && first_osc_all > 1
    xline(z_star_values(first_osc_all), 'k--', 'LineWidth', 2, ...
          'Label', 'Hopf Bifurcation', 'LabelOrientation', 'horizontal');
end

% Add text annotation with bifurcation analysis
xlim([min(z_star_values)-2, max(z_star_values)+2]);
ylim_orig = ylim;
if ~isempty(first_osc_all)
    text(min(z_star_values)+2, ylim_orig(2)*0.9, ...
        sprintf('Stable Steady State (z^* < %d) → Limit Cycle Oscillations (z^* ≥ %d)', ...
        z_star_values(first_osc_all)-10, z_star_values(first_osc_all)), ...
        'FontSize', 11, 'BackgroundColor', 'w', 'EdgeColor', 'k');
end

% Print oscillation summary
fprintf('\n========================================\n');
fprintf('OSCILLATION SUMMARY:\n');
fprintf('========================================\n');
fprintf('z^*\tX_osc\tY_osc\tZ_osc\tPeriod_X\tPeriod_Y\tPeriod_Z\n');
for i = 1:length(z_star_values)
    fprintf('%d\t%d\t%d\t%d\t%.1f\t\t%.1f\t\t%.1f\n', ...
        z_star_values(i), osc_x(i), osc_y(i), osc_z(i), ...
        per_x(i), per_y(i), per_z(i));
end

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
fid = fopen(fullfile(export_dir, 'amplitude_vs_zstar_YZ_PS.dat'), 'w');
fprintf(fid, 'z_star\tamp_x\tamp_y\tamp_z\n');
for i = 1:length(z_star_values)
    fprintf(fid, '%.1f\t%.6f\t%.6f\t%.6f\n', z_star_values(i), amp_x(i), amp_y(i), amp_z(i));
end
fclose(fid);
fprintf('Exported: amplitude_vs_zstar_YZ_PS.dat\n');

% 2. Export period vs z_star data
fid = fopen(fullfile(export_dir, 'period_vs_zstar_YZ_PS.dat'), 'w');
fprintf(fid, 'z_star\tper_x\tper_y\tper_z\n');
for i = 1:length(z_star_values)
    fprintf(fid, '%.1f\t%.6f\t%.6f\t%.6f\n', z_star_values(i), per_x(i), per_y(i), per_z(i));
end
fclose(fid);
fprintf('Exported: period_vs_zstar_YZ_PS.dat\n');

% 3. Export bifurcation data - separate files for each species
% X species
fid = fopen(fullfile(export_dir, 'x_branch_YZ_PS.dat'), 'w');
fprintf(fid, 'z_star\tx_max\tx_min\n');
for i = 1:length(z_star_values)
    if ~isnan(bifurcation_x_max(i))
        fprintf(fid, '%.1f\t%.6f\t%.6f\n', z_star_values(i), bifurcation_x_max(i), bifurcation_x_min(i));
    end
end
fclose(fid);
fprintf('Exported: x_branch_YZ_PS.dat\n');

% Y species
fid = fopen(fullfile(export_dir, 'y_branch_YZ_PS.dat'), 'w');
fprintf(fid, 'z_star\ty_max\ty_min\n');
for i = 1:length(z_star_values)
    if ~isnan(bifurcation_y_max(i))
        fprintf(fid, '%.1f\t%.6f\t%.6f\n', z_star_values(i), bifurcation_y_max(i), bifurcation_y_min(i));
    end
end
fclose(fid);
fprintf('Exported: y_branch_YZ_PS.dat\n');

% Z species
fid = fopen(fullfile(export_dir, 'z_branch_YZ_PS.dat'), 'w');
fprintf(fid, 'z_star\tz_max\tz_min\n');
for i = 1:length(z_star_values)
    if ~isnan(bifurcation_z_max(i))
        fprintf(fid, '%.1f\t%.6f\t%.6f\n', z_star_values(i), bifurcation_z_max(i), bifurcation_z_min(i));
    end
end
fclose(fid);
fprintf('Exported: z_branch_YZ_PS.dat\n');

% 4. Export time series data for demo values
for zs = demo_vals
    key = sprintf('z%d', zs);
    if isfield(demo_store, key)
        S = demo_store.(key);
        t = S.t;
        y = S.y;
        
        filename = fullfile(export_dir, sprintf('timeseries_z%d_YZ_PS.dat', zs));
        fid = fopen(filename, 'w');
        fprintf(fid, 'time\tx_dilt\ty_dilt\tz_dilt\n');
        
        % Decimate to ~1000 points for manageable file size
        step = max(1, floor(length(t)/1000));
        for i = 1:step:length(t)
            fprintf(fid, '%.6f\t%.6f\t%.6f\t%.6f\n', ...
                t(i), y(i,1), y(i,2), y(i,4));
        end
        fclose(fid);
        fprintf('Exported: timeseries_z%d_YZ_PS.dat\n', zs);
    end
end

% =============================================================================
% ========================= LOCAL FUNCTIONS ===================================
% =============================================================================

function dydt = repressilator_rhs_YZ_only(~, y, p)
    % YZ PHASE SEPARATION ONLY VERSION: Only Y and Z phase separate
    % X does NOT participate in phase separation (no dense phase)
    % State vector: [x_dilt; y_dilt; y_dens; z_dilt; z_dens]
    
    % State variables - ensure non-negative
    x_dilt = max(y(1), 0);
    y_dilt = max(y(2), 0);
    y_dens = max(y(3), 0);
    z_dilt = max(y(4), 0);
    z_dens = max(y(5), 0);

    % Total populations for phase-separating species
    y_bar = y_dilt + y_dens;
    z_bar = z_dilt + z_dens;

    % Production terms (repression)
    prod_x = p.alpha_x / (1 + (z_dilt / p.K_zx)^p.n);
    prod_y = p.alpha_y / (1 + (x_dilt / p.K_xy)^p.n);
    prod_z = p.alpha_z / (1 + (y_dilt / p.K_yz)^p.n);

    % Initialize phase separation rates
    k_in_y = 0; k_out_y = 0;
    k_in_z = 0; k_out_z = 0;

    % ---- Phase separation in Y only ----
    if (y_dilt >= p.y_star || y_dens > 0)
        deltaF_y = free_energy_local(y_dilt + 1, y_bar, p.mu_y, p.thermal_energy, p.volume_factor) ...
                 - free_energy_local(y_dilt, y_bar, p.mu_y, p.thermal_energy, p.volume_factor);
        
        k_in_y = (6 * p.D * y_dilt) / ((p.Vtot - p.v * y_dens)^(2/3) + eps);
        
        if y_dens > 0
            k_out_y = ((6 * p.D * (y_dilt+1)) / ((p.Vtot - p.v * (y_dens-1))^(2/3) + eps)) * exp(-deltaF_y / p.thermal_energy);
        end
    end

    % ---- Phase separation in Z only ----
    if (z_dilt >= p.z_star || z_dens > 0)
        deltaF_z = free_energy_local(z_dilt + 1, z_bar, p.mu_z, p.thermal_energy, p.volume_factor) ...
                 - free_energy_local(z_dilt, z_bar, p.mu_z, p.thermal_energy, p.volume_factor);
        
        k_in_z = (6 * p.D * z_dilt) / ((p.Vtot - p.v * z_dens)^(2/3) + eps);
        
        if z_dens > 0
            k_out_z = ((6 * p.D * (z_dilt+1)) / ((p.Vtot - p.v * (z_dens-1))^(2/3) + eps)) * exp(-deltaF_z / p.thermal_energy);
        end
    end

    % Dynamics (X has no phase separation terms)
    dxdilt_dt = prod_x - p.b_x * x_dilt;

    dydilt_dt = prod_y - p.b_y * y_dilt - k_in_y + k_out_y;
    dydens_dt = k_in_y - k_out_y - p.b_y * y_dens;

    dzdilt_dt = prod_z - p.b_z * z_dilt - k_in_z + k_out_z;
    dzdens_dt = k_in_z - k_out_z - p.b_z * z_dens;

    % Return derivatives in same order as state vector
    dydt = [dxdilt_dt; dydilt_dt; dydens_dt; dzdilt_dt; dzdens_dt];
end

function F = free_energy_local(dilt, bar, mu_local, thermal_energy, volume_factor)
    epsv = 1e-12;
    denom = max(volume_factor - (bar - dilt), epsv);
    temp1 = dilt / denom;
    temp1 = max(temp1, epsv);
    term1 = -mu_local * (bar - dilt);
    term2 = thermal_energy * dilt * (log(temp1) - 1);
    F = term1 + term2;
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

function [amp, period, is_oscillatory] = robust_oscillation_analysis(t, s, min_cycles_for_analysis)
    % ROBUST_OSCILLATION_ANALYSIS - Comprehensive oscillation characterization
    %
    % Inputs:
    %   t - time vector
    %   s - signal vector
    %   min_cycles_for_analysis - minimum number of cycles to consider (default: 3)
    %
    % Outputs:
    %   amp - oscillation amplitude (0 if not oscillatory)
    %   period - oscillation period (NaN if not oscillatory)
    %   is_oscillatory - boolean indicating if sustained oscillations are present
    
    if nargin < 3
        min_cycles_for_analysis = 3;
    end
    
    s = s(:);
    t = t(:);
    
    % Default outputs for non-oscillatory case
    amp = 0;
    period = NaN;
    is_oscillatory = false;
    
    % Remove any NaN or Inf
    valid_idx = isfinite(s) & isfinite(t);
    s = s(valid_idx);
    t = t(valid_idx);
    
    if length(s) < 100  % Need sufficient data
        return;
    end
    
    % Detrend to remove slow drift
    p = polyfit(t, s, 1);
    s_detrended = s - polyval(p, t);
    
    % Initialize metrics
    period_acf = NaN;
    period_fft = NaN;
    period_zc = NaN;
    cv_zc = inf;
    pks_acf = [];
    locs_acf = [];
    
    % Method 1: Autocorrelation analysis (with error handling)
    try
        [acf, lags] = xcorr(s_detrended - mean(s_detrended), 'coeff');
        lags = lags * (t(2)-t(1));  % Convert to time units
        
        % Find peaks in autocorrelation (positive lags only)
        half_len = floor(length(lags)/2);
        acf_pos = acf(half_len+1:end);
        lags_pos = lags(half_len+1:end);
        
        % Find significant peaks in ACF (with try-catch for empty results)
        if max(acf_pos) > 0.3
            [pks_acf, locs_acf] = findpeaks(acf_pos, 'MinPeakHeight', 0.3, ...
                                                   'MinPeakDistance', 5);
            if ~isempty(pks_acf)
                period_acf = lags_pos(locs_acf(1));
            end
        end
    catch
        % Autocorrelation failed - continue with other methods
    end
    
    % Method 2: FFT-based period estimation (with error handling)
    try
        fs = 1/mean(diff(t));
        [pxx, f] = pwelch(s_detrended, [], [], [], fs);
        
        % Find dominant frequency (excluding very low frequencies)
        valid_f_idx = f > 1/(max(t)-min(t)) & f < fs/2;
        if sum(valid_f_idx) > 0
            [max_pxx, max_f_idx] = max(pxx(valid_f_idx));
            if max_pxx > 0
                f_dom = f(valid_f_idx);
                f_dom = f_dom(max_f_idx);
                period_fft = 1/f_dom;
            end
        end
    catch
        % FFT failed - continue
    end
    
    % Method 3: Zero-crossing analysis (with error handling)
    try
        zero_crossings = find(diff(sign(s_detrended)) ~= 0);
        if length(zero_crossings) >= 4
            crossing_times = t(zero_crossings);
            half_periods = diff(crossing_times);
            periods_zc = 2 * half_periods(1:2:end);
            periods_zc = periods_zc(periods_zc > 0);
            if ~isempty(periods_zc)
                period_zc = median(periods_zc);
                % Check consistency of zero-crossing periods
                if mean(periods_zc) > 0
                    cv_zc = std(periods_zc) / mean(periods_zc);  % Coefficient of variation
                else
                    cv_zc = inf;
                end
            end
        end
    catch
        % Zero-crossing failed - continue
    end
    
    % Method 4: Peak-to-peak analysis with robust peak detection
    peak_sets = {};
    valley_sets = {};
    
    % Only attempt peak detection if signal has sufficient variation
    if max(s_detrended) - min(s_detrended) > 1e-6
        try
            prom_thresholds = [0.1, 0.15, 0.2, 0.25] * (max(s)-min(s));
            
            for p_idx = 1:length(prom_thresholds)
                [pks_curr, locs_curr] = findpeaks(s_detrended, 'MinPeakProminence', prom_thresholds(p_idx));
                [vally_curr, locs_valley] = findpeaks(-s_detrended, 'MinPeakProminence', prom_thresholds(p_idx));
                
                if length(locs_curr) >= min_cycles_for_analysis
                    peak_sets{end+1} = struct('peaks', pks_curr, 'locs', locs_curr);
                end
                if length(locs_valley) >= min_cycles_for_analysis
                    valley_sets{end+1} = struct('valleys', -s_detrended(locs_valley), 'locs', locs_valley);
                end
            end
        catch
            % Peak detection failed - continue
        end
    end
    
    % Choose the most consistent peak set
    best_peak_set = [];
    best_peak_consistency = inf;
    
    for i = 1:length(peak_sets)
        locs = peak_sets{i}.locs;
        if length(locs) >= min_cycles_for_analysis
            intervals = diff(t(locs));
            if ~isempty(intervals) && mean(intervals) > 0
                cv = std(intervals) / mean(intervals);  % Coefficient of variation
                if cv < best_peak_consistency
                    best_peak_consistency = cv;
                    best_peak_set = peak_sets{i};
                end
            end
        end
    end
    
    % Determine if system is oscillatory based on multiple criteria
    criteria_met = 0;
    total_criteria = 0;
    
    % Criterion 1: Clear peak in autocorrelation
    total_criteria = total_criteria + 1;
    if ~isempty(pks_acf) && max(pks_acf) > 0.4
        criteria_met = criteria_met + 1;
    end
    
    % Criterion 2: Consistent FFT peak
    total_criteria = total_criteria + 1;
    if ~isnan(period_fft) && period_fft > 0 && period_fft < (max(t)-min(t))/2
        criteria_met = criteria_met + 1;
    end
    
    % Criterion 3: Consistent zero-crossings
    total_criteria = total_criteria + 1;
    if ~isnan(period_zc) && cv_zc < 0.3  % Less than 30% variation
        criteria_met = criteria_met + 1;
    end
    
    % Criterion 4: At least 3 complete cycles detected with good consistency
    total_criteria = total_criteria + 1;
    if ~isempty(best_peak_set) && length(best_peak_set.locs) >= min_cycles_for_analysis && best_peak_consistency < 0.3
        criteria_met = criteria_met + 1;
    end
    
    % Decide if oscillatory (at least 3 out of 4 criteria met, or 2 out of 3 if some methods failed)
    if total_criteria > 0
        is_oscillatory = (criteria_met >= max(2, total_criteria - 1));
    else
        is_oscillatory = false;
    end
    
    if is_oscillatory && ~isempty(best_peak_set)
        % Calculate robust amplitude and period
        
        % Use median of peak-to-valley differences for amplitude
        peak_times = t(best_peak_set.locs);
        peak_values = best_peak_set.peaks;
        
        % Find corresponding valleys
        if ~isempty(valley_sets)
            % Use the most consistent valley set (simplified - use first with sufficient points)
            best_valley_set = [];
            for i = 1:length(valley_sets)
                if length(valley_sets{i}.locs) >= min_cycles_for_analysis
                    best_valley_set = valley_sets{i};
                    break;
                end
            end
            
            if ~isempty(best_valley_set)
                valley_times = t(best_valley_set.locs);
                valley_values = -best_valley_set.valleys;  % Convert back to actual values
                
                % Match peaks with nearest valleys
                amplitudes = [];
                for j = 1:min(length(peak_values), 10)  % Limit to first 10 cycles
                    if j <= length(valley_times)
                        amplitudes(end+1) = abs(peak_values(j) - valley_values(j));
                    end
                end
                if ~isempty(amplitudes)
                    amp = median(amplitudes);
                else
                    % Fallback: use half of peak-to-peak range
                    amp = 0.5 * (max(s_detrended) - min(s_detrended));
                end
            else
                % Fallback if valleys not detected
                amp = 0.5 * (max(s_detrended) - min(s_detrended));
            end
        else
            % Fallback if valleys not detected
            amp = 0.5 * (max(s_detrended) - min(s_detrended));
        end
        
        % Period calculation: use median of intervals from multiple methods
        periods = [];
        
        % From peaks
        if length(peak_times) >= min_cycles_for_analysis
            peak_intervals = diff(peak_times);
            periods = [periods; peak_intervals(:)];
        end
        
        % From autocorrelation if available and reasonable
        if ~isnan(period_acf) && period_acf > 0 && period_acf < (max(t)-min(t))/2
            periods = [periods; period_acf];
        end
        
        % From FFT if available and reasonable
        if ~isnan(period_fft) && period_fft > 0 && period_fft < (max(t)-min(t))/2
            periods = [periods; period_fft];
        end
        
        % From zero-crossings if available and reasonable
        if ~isnan(period_zc) && period_zc > 0 && period_zc < (max(t)-min(t))/2
            periods = [periods; period_zc];
        end
        
        % Take robust median of all available period estimates
        if ~isempty(periods)
            period = median(periods);
            
            % Sanity check: period should be positive and reasonable
            if period <= 0 || period > (max(t)-min(t))/2
                period = NaN;
                is_oscillatory = false;
            end
        else
            period = NaN;
            is_oscillatory = false;
        end
    end
    
    % Final threshold: amplitude must be significant
    signal_range = max(s) - min(s);
    if signal_range > 0 && amp < 0.01 * signal_range
        amp = 0;
        is_oscillatory = false;
    end
    
    % If not oscillatory, ensure period is NaN
    if ~is_oscillatory
        period = NaN;
    end
end