% -------------------------------------------------------------------------
% Script for simulating a plate reverb.
%
% This plate reverb model uses modal synthesis approach and includes:
% 1. Second-order accurate and exact difference schemes.
% 2. Modal pruning.
% 3. Custom T60 approximation.
% 4. Saturation effect.
% 5. Reverb dry/wet and pre-delay controls.
% 6. Stereo output.
%
% There are three input forcing options for the algorithm:
% 1. "impulse" (dirac delta function).
% 2. "sweep" (sine sweeping technique to obtain impulse response).
% 3. "file" (to apply reverb to a recording).
%
% NB: there is no built-in resampling in a script so the sample rate of an 
% input file should match the simulation sample rate. Otherwise the input
% will be changed to sweep.
%
% References:
% [1] Stefan Bilbao. Numerical Sound Synthesis: Finite Difference Schemes 
%     and Simulation in Musical Acoustics.
%     https://doi.org/10.1002/9780470749012
% [2] M. Ducceschi and C. J. Webb. Plate Reverberation: Towards the 
%     Development of a Real-Time Plug-In for the Working Musician. Proc. 
%     ICA 2016.
%     http://mdphys.org/PDF/icaPLA_2016.pdf
%
% Author: Victor Zheleznov
% Date: 20/03/2023
% -------------------------------------------------------------------------

clc; close all; clear all;

% sample rate
SR = 48e3; % [Hz]

% flags
USE_EXACT = true;   % use exact update scheme [1]
USE_PRUNING = true; % apply modal pruning

% physical parameters
Lx = 2;       % dimensions [m]
Ly = 1;       %
H = 5e-4;     % thickness [m]
T = 700;      % tension [N/m]
rho = 7.87e3; % density [kg/m^3]
E = 200e9;    % youngs modulus [N/m^2]
v = 0.29;     % poisons ratio

% input/output
pos_in = [0.5, 0.5];            % normalised input point (0-1)
pos_out = [0.2, 0.6; 0.4, 0.2]; % normalised output points (0-1)
input = "sweep";                % impulse, sweep or file
file_path = "tabla_loop.wav";
sweep_dur = 0.5;                % sweep duration [sec]

% pruning
tol_pr = 1e-6; % discard modes with modal shape function close to zero at input/output points
ncent = 0.1;   % discard modes based on this cents distance

% reverb controls 
rev_drywet = 0.3; % reverb dry/wet (0-1)
sat_drywet = 0.2; % saturation dry/wet (0-1)
sat_amp = 5;      % saturation gain
Tpd = 50e-3;      % pre-delay [sec]

% T60
T60_arr = [7,  7,   5,   8,   5,   4,   3,   2,   1   ]; % T60 values [sec]
f_arr =   [63, 125, 250, 500, 1e3, 2e3, 4e3, 8e3, 16e3]; % corresponding frequency bands [Hz]

% check parameters
assert(SR > 0, "Sample rate must be positive!");
assert(Lx > 0 && Ly > 0, "Plate dimensions must be positive!");
assert(H > 0, "Plate thickness must be positive!");
assert(T > 0, "Tension must be positive!");
assert(rho > 0, "Density must be positive!");
assert(E > 0, "Youngs modulus must be positive!");
assert(min((pos_in > 0) & (pos_in < 1) & min((pos_out > 0) & (pos_out < 1))) == 1,... 
       "Normalised input/output points must be between 0 and 1!");
assert(min(T60_arr) > 0, "T60 times must be positive!");
assert(min(f_arr) > 0 & max(f_arr) < SR/2, "Frequency bands for T60 approximation should be within (0,SR/2)!");
assert(tol_pr > 0 & ncent > 0, "Modal pruning parameters must be positive!");
assert((rev_drywet >= 0 & rev_drywet <= 1) & (sat_drywet >= 0 & sat_drywet <= 1),... 
       "Dry/wet parameters should be in [0,1]!");
assert(Tpd >= 0, "Pre-delay time should be non-negative!");
assert(sweep_dur > 0, "Sweep duration should be positive!");

% derived parameters
k = 1/SR;                         % time step
K = sqrt(E*H^2/(12*rho*(1-v^2))); % stiffness
c = sqrt(T/(rho*H));              % wavespeed
Nf = floor(max(T60_arr)*SR);      % simulation duration in samples
Npd = floor(Tpd*SR);              % pre-delay duration in samples

% generate reverb input
if strcmp(input, "impulse") == 1
    in = [1; zeros(Nf-1,1)];
elseif strcmp(input, "sweep") == 1
    [in, in_f] = gen_sine_sweep(sweep_dur, max(T60_arr), SR, 20, SR/2, 0.95, 1e-2, false);
elseif strcmp(input, "file") == 1
    [in, inSR] = audioread(file_path);
    if size(in,2) == 2
        in = mean(in,2);
    end
    if inSR == SR
        in = [in; zeros(Nf,1)]; % add zeros for T60
        Nf = length(in);        % account for file size in simulation duration
    else
        warning("Input file has a different sample rate! Changed input to sweep");
        input = "sweep";
        [in, in_f] = gen_sine_sweep(sweep_dur, max(T60_arr), SR, 20, SR/2, 0.95, 1e-2, false);
    end
else
    error("Input should be set to impulse, sweep or file!");
end

% derive maximum frequency and wavenumber from stability condition
if USE_EXACT == true
    wmax = pi*SR;
else
    wmax = 2/k;
end
bmax = sqrt(-(c^2/(2*K^2)) + sqrt((c^2/(2*K^2))^2 + wmax^2/K^2));
Mx = floor(sqrt(bmax^2*Lx^2/pi^2 - (Lx/Ly)^2));
My = floor(sqrt(bmax^2*Ly^2/pi^2 - (Ly/Lx)^2));

% create modal index pairs
[mx,my] = meshgrid(1:Mx, 1:My);
mx = mx(:);
my = my(:);

% calculate wavenumbers
beta = sqrt((mx*pi/Lx).^2 + (my*pi/Ly).^2);
mx(beta > bmax) = [];
my(beta > bmax) = [];
beta(beta > bmax) = [];

% calculate modal frequencies
omega = sqrt(c^2*beta.^2 + K^2*beta.^4);

% calculate damping coefficients
sigma = calc_sigma(omega, T60_arr, f_arr, SR);
assert(min(sigma > 0), "Damping coefficients should be positive! Check specified T60 values...");

% calculate modal shape functions
Phi_in = Phi(pos_in(1), pos_in(2), mx, my, Lx, Ly);
Phi_outL = Phi(pos_out(1,1), pos_out(1,2), mx, my, Lx, Ly);
Phi_outR = Phi(pos_out(2,1), pos_out(2,2), mx, my, Lx, Ly);

% modes pruning
if USE_PRUNING == true
    Mst = length(mx);
    
    % find modal shape functions close to zero in input/output points
    pr_idx_in = find(abs(Phi_in) < tol_pr);
    pr_idx_outL = find(abs(Phi_outL) < tol_pr);
    pr_idx_outR = find(abs(Phi_outR) < tol_pr);
    pr_idx = union(pr_idx_in, intersect(pr_idx_outL, pr_idx_outR));
    
    % apply pruning
    mx(pr_idx) = [];
    my(pr_idx) = [];
    omega(pr_idx) = [];
    sigma(pr_idx) = [];
    Phi_in(pr_idx) = [];
    Phi_outL(pr_idx) = [];
    Phi_outR(pr_idx) = [];
    Mpr1 = length(mx); 
    
    % pruning based on cents distance [2]
    pr_idx = [];
    freq = omega/(2*pi);
    if ncent > 0
        while min(freq) ~= Inf
            [f_fix, idx_fix] = min(freq);
            pr_idx_cur = find(1200*log2(freq/f_fix) < ncent);
            freq(pr_idx_cur) = Inf;
            if length(pr_idx_cur) > 1
                pr_idx_cur(pr_idx_cur == idx_fix) = [];
                pr_idx = [pr_idx; pr_idx_cur];
            end
        end
    end
    
    % apply pruning
    mx(pr_idx) = [];
    my(pr_idx) = [];
    omega(pr_idx) = [];
    sigma(pr_idx) = [];
    Phi_in(pr_idx) = [];
    Phi_outL(pr_idx) = [];
    Phi_outR(pr_idx) = [];
    Mpr2 = length(mx);
    
    disp("Pruning: " + num2str(Mst) + " -> " + num2str(Mpr1) + " -> " + num2str(Mpr2) + " modes!");
end

% define output
out = zeros(Nf,2);

% allocate memory
M = length(mx);
p = zeros(M,1);
p1 = zeros(M,1);
p2 = zeros(M,1);

% precompute update matrices for loop
if USE_EXACT == true
    cond = (sigma <= omega);
    idx = find(cond == 0);
    B = 2*exp(-sigma*k).*cos(sqrt(omega.^2-sigma.^2)*k);
    B(idx) = exp(-sigma(idx)*k).*(exp(sqrt(sigma(idx).^2-omega(idx).^2)*k) + exp(-sqrt(sigma(idx).^2-omega(idx).^2)*k));
    C = -exp(-2*sigma*k);
else
    B = (2-k^2*omega.^2)./(1+k*sigma);
    C = (k*sigma-1)./(1+k*sigma);
end
J = (k^2*Phi_in/(rho*H))./(1+k*sigma);

% main loop
tic
in = in./max(abs(in));
in_sat = (1 - sat_drywet)*in + sat_drywet*tanh(sat_amp*in);
for n = 1:Nf
    % update state
    p = B.*p1 + C.*p2 + J*in_sat(n);
    out(n,1) = Phi_outL.'*p;
    out(n,2) = Phi_outR.'*p;
    % shift state
    p2 = p1;
    p1 = p;
end
exc_time = toc;
disp("Audio length = " + sprintf("%.2f sec.", Nf/SR))
disp("Execution time = " + sprintf("%.2f sec.", exc_time))

% process output
if strcmp(input, "impulse") == 1
    ir = diff(out);
    ir = ir./max(abs(ir),[],'all');
    fig_spec = myspec(ir(:,1), SR, 512, 0.75);
    soundsc(ir, SR);
    audiowrite('plate_reverb_ir.wav', ir, SR);
elseif strcmp(input, "sweep") == 1
    ir = [conv(out(:,1), in_f), conv(out(:,2), in_f)];
    ir(1:length(in_f),:) = [];
    ir = diff(ir);
    ir = ir./max(abs(ir),[],'all');
    fig_spec = myspec(ir(:,1), SR, 512, 0.75);
    clim = caxis;
    caxis([clim(2)-120, clim(2)]);
    soundsc(ir, SR);
    audiowrite('plate_reverb_ir.wav', ir, SR);
elseif strcmp(input, "file") == 1
    % normalise and differentiate
    out_diff = diff(out);
    out_diff = out_diff./max(abs(out_diff),[],'all');
    in = in(1:length(out_diff));
    in = [in, in];

    % apply pre-delay
    out_diff = [zeros(Npd,2); out_diff];
    in = [in; zeros(Npd,2)];

    % apply dry/wet
    out_diff_drywet = (1 - rev_drywet)*in + rev_drywet*out_diff;
    
    % listen and save
    soundsc(out_diff_drywet, SR);
    [~,file_name] = fileparts(file_path);
    audiowrite(append(file_name, '_plate_reverb', '.wav'), out_diff_drywet, SR);
end

%% FUNCTIONS
% calculate modal functions
% input:
%   x, y --- normalised 2D coordinates;
%   mx, my --- modal index pairs;
%   Lx, Ly --- dimensions.
% output:
%   z --- modal functions evaluated at x,y.
function z = Phi(x, y, mx, my, Lx, Ly)
    z = (2/sqrt(Lx*Ly))*sin(mx*pi*x).*sin(my*pi*y);
end

% calculate loss parameters
% input:
%   omega --- modal angular frequencies [rad];
%   T60_arr --- T60 values [sec];
%   f_arr --- corresponding frequency bands [Hz];
%   SR --- sample rate [Hz];
% output:
%   sigma --- loss parameters vector.
function sigma = calc_sigma(omega, T60_arr, f_arr, SR)
    % check input size
    if size(T60_arr,1) ~= 1
        T60_arr = T60_arr.';
    end
    if size(f_arr,1) ~= 1
        f_arr = f_arr.';
    end
    if size(omega,2) ~= 1
        omega = omega.';
    end
    
    % sort input
    [f_arr, sort_idx] = sort(f_arr);
    T60_arr = T60_arr(sort_idx);
    
    % add boundary
    f_arr = [1e-6, f_arr, SR/2];
    T60_arr = [T60_arr(1), T60_arr, T60_arr(end)];
    
    % calculate linear approximation (based on cents distance)
    f = omega/(2*pi);
    ineq = (f >= f_arr);
    f = f.';
    idx = sum(ineq,2);
    sigma_arr = (6*log(10)./T60_arr).';
    w = (log2(f./f_arr(idx)) ./ log2(f_arr(idx+1)./f_arr(idx))).';
    sigma = sigma_arr(idx).*(1-w) + sigma_arr(idx+1).*w;
end

% generate logarithmic sine sweep
% input:
%   t_dur --- sweep duration [sec];
%   t_sil --- silence duration [sec];
%   fs --- sample rate [Hz];
%   f0 --- lowest sweep frequency [Hz];
%   f1 --- highest sweep frequency [Hz];
%   max_amp --- maximum amplitude of the sweep;
%   t_fade --- fade in/out duration [sec];
%   PLOT_SPECTRUM --- flag to plot spectrum of sweep and inverse filter.
% output:
%   x --- sweep signal;
%   f --- inverse filter.
function [x,f] = gen_sine_sweep(t_dur, t_sil, fs, f0, f1, max_amp, t_fade, PLOT_SPECTRUM)
    t = (0:1/fs:t_dur).';

    R = log(f1/f0); % sweep rate
    sweep_arg = 2*pi*(f0*t_dur/R)*(exp(t*R/t_dur)-1);
    x = max_amp*sin(sweep_arg);
    len_x = length(x);

    % generate fade-in/out
    len_fade = t_fade*fs;
    fade = 0.5*(1 - cos(2*pi.*(0:len_fade-1)./(2*len_fade))).';
    fade = [fade; ones(len_x-2*len_fade, 1); flipud(fade)];
    x = x.*fade;

    % generate inverse filter
    f = flipud(x)./exp(t*R/t_dur);
    
    % add silence
    Nsil = floor(fs*t_sil);
    x = [x; zeros(Nsil,1)];

    % plot spectrum
    if PLOT_SPECTRUM
        NFFT = 2^(ceil(log(len_x)/log(2))); % next power of 2
        NFFT_2 = NFFT/2 + 1;
        X = fft(x, NFFT); X = X(1:NFFT_2);
        F = fft(f, NFFT); F = F(1:NFFT_2);

        fig_fft = figure;
        freq = (0:fs/NFFT:fs/2).';
        hold on;
        plot(freq, 20*log10(abs(X)), 'r');
        plot(freq, 20*log10(abs(F)), 'b');
        xlabel('Frequency [Hz]', 'interpreter', 'latex');
        ylabel('Magnitude [dB]', 'interpreter', 'latex');
        title('Spectogram for the sine sweep', 'interpreter', 'latex');
        leg = legend({'Sine sweep', 'Inverse filter'}, 'interpreter', 'latex');
        set(leg,'location','southeast')

        myspec(x, fs, 512, 0.95);
    end
end

% create a spectogram plot of an input signal
% input:
%   x - mono input signal;
%   Fs - sampling frequency [Hz];
%   N - frame length;
%   O - overlap factor (between 0 and 1).
function fig_spec = myspec(x, Fs, N, O)
    % find hop size
    HA = round(N - O*N);

    % generate window
    win = 0.5*(1 - cos(2*pi.*(0:N-1)./N)).';

    % calculate number of frames
    L = length(x);
    NF = ceil(L/HA);
    x = [x; zeros((NF-1)*HA+N-L,1)];
    
    % STFT size
    NFFT = 2^(ceil(log(N)/log(2))); % next power of 2
    NFFT_2 = NFFT / 2 + 1;

    % calculate STFT
    STFT = zeros(NFFT_2, NF);
    for m = 0:NF-1
        x_frame = win.*x((1:N).'+m*HA);
        X = fft(x_frame, NFFT);
        STFT(:,m+1) = X(1:NFFT_2);
    end
    
    % plot spectogram
    fig_spec = figure;
    t = ((0:NF-1).*HA/Fs).';
    freq = (0:Fs/NFFT:Fs/2).';
    STFT_dB = 20*log10(abs(STFT));
    max_dB = max(max(STFT_dB));
    imagesc(t, freq, STFT_dB, 'CDataMapping', 'scaled');
    c = colorbar;
    c.Label.String = 'dB';
    colormap hot
    caxis([max_dB-60, max_dB]);
    xlim([0 t(end)]);
    ylim([0 freq(end)]);
    ax_spec = fig_spec.CurrentAxes;
    set(ax_spec, 'YDir', 'normal');
    set(ax_spec, 'YTick', 0:1000:Fs/2);
    set(ax_spec, 'YTickLabel', 0:1000:Fs/2);
    xlabel('Time [s]', 'interpreter', 'latex');
    ylabel('Frequency [Hz]', 'interpreter', 'latex');
    title_str = sprintf("Spectogram with frame length = $%d$ ms and overlap factor = $%d$\\%%", floor((N/Fs)*1e3), O*1e2);
    title(title_str, 'interpreter', 'latex');
end