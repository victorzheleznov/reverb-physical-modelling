% -------------------------------------------------------------------------
% Script for simulation a spring reverb effect.
%
% This script considers a simplified helical spring model [1,2] for 
% simulating a reverb effect. Boundary conditions are chosed to be simply
% supported for transverse displacement and fixed for longitudinal
% displacement. The external forcing of the spring is done by incorporating
% a force term into helical spring equations [3].
%
% There are three input forcing options for an algorithm:
% 1. "impulse" (dirac delta function).
% 2. "sweep" (sine sweeping technique to obtain impulse response).
% 3. "file" (to apply reverb to a recording).
%
% There are three options for the output signal:
% 1. "displacement" (displacement of the last free spring point);
% 2. "velocity" (velocity of the last free spring point);
% 3. "force" (force exerted by the spring on the last free point).
% The last two options produce high-passed impulse responses which doesn't
% correspond well to the real string. Thus displacement was chosen as the
% default output signal.
%
% NB: since simulation may take several minutes a demo impulse response
% "spring_reverb_ir.wav" is included in the repository. This recording
% corresponds to the parameters set in this script.
%
% NB: the script doesn't have built-in optimisation procedure to obtain 
% free parameters of numerical method. Spring parameters in dimensionless 
% form and free numerical method parameters are taken from the article [1].
%
% NB: there is no built-in resampling in a script so the sample rate of an 
% input file should match the simulation sample rate. Otherwise the input
% will be changed to sweep.
%
% References:
% [1] Bilbao, S., Parker, J. (2010). A virtual model of spring 
%     reverberation. IEEE Transactions on Audio, Speech and Language 
%     Processing, 18(4), 799-808. 
%     https://doi.org/10.1109/TASL.2009.2031506
% [2] Stefan Bilbao. Numerical Sound Synthesis: Finite Difference Schemes 
%     and Simulation in Musical Acoustics.
%     https://doi.org/10.1002/9780470749012
% [3] Van Walstijn, M. (2020). Numerical Calculation of Modal Spring Reverb 
%     Parameters. In 23rd International Conference on Digital Audio 
%     Effects: (online) (pp. 38-45).
%     https://dafx2020.mdw.ac.at/
%
% Author: Victor Zheleznov
% Date: 23/03/2023
% -------------------------------------------------------------------------

clc; close all; clear all;

% simulation parameters
SR = 44.1e3;               % sample rate [Hz]
Tf = 4;                    % duration [sec]
out_type = "displacement"; % output signal type: displacement, velocity or force

% physical parameters [1]
K = 0.0246;     
q = 1905;
gamma = 1190;
sigma_t = 1.65; % longitudinal damping parameter
sigma_l = 1.65; % transversal damping parameter

% numerical scheme parameters [1]
alpha = 0.5;
eta = 0.09;
theta = 0.00022;
heps = 1e-6;     % grid step for finding minimal value

% forcing parameters
input = "sweep";     % impulse, sweep or file
file_path = "tabla_loop.wav";
th_exc = pi/2;       % excitation angle [rad]
th_pick = pi/2;      % pick-up angle [rad]
perc_sweep = 0.25;   % percentage of simulation duration for sweep
perc_silence = 0.75; % percentage of simulation duration for silence

% reverb controls
rev_drywet = 0.3; % reverb dry/wet (0-1)

% check parameters
assert(SR > 0, "Sample rate must be positive!");
assert(Tf > 0, "Simulation duration should be positive!");
assert(K > 0 & q > 0 & gamma > 0, "Physical parameters should be positive!");
assert(sigma_t > 0 & sigma_l > 0, "Damping parameters should be positive!");
assert(heps > 0, "Grid step should be positive!");
assert(perc_sweep > 0 & perc_sweep <= 1, "Sweep duration percentage should be in (0,1]!");
assert(perc_silence >= 0 & perc_silence < 1, "Silence duration percentage should be in [0,1)!");
assert(rev_drywet >= 0 & rev_drywet <= 1, "Dry/wet parameter should be in [0,1]!");

% derived parameters
k = 1/SR;          % time step [sec]
Nf = floor(Tf*SR); % number of samples

% generate reverb input
if strcmp(input, "impulse") == 1
    in = [1; zeros(Nf-1,1)];
elseif strcmp(input, "sweep") == 1
    [in, in_f] = gen_sine_sweep(perc_sweep*Tf, perc_silence*Tf, SR, 20, SR/2, 0.95, 1e-2, false);
elseif strcmp(input, "file") == 1
    [in, inSR] = audioread(file_path);
    if size(in,2) == 2
        in = mean(in,2);
    end
    if inSR == SR
        in = [in; zeros(Nf,1)]; % add zeros for decay
        Nf = length(in);        % account for file size in simulation duration
    else
        warning("Input file has a different sample rate! Changed input to sweep");
        input = "sweep";
        [in, in_f] = gen_sine_sweep(perc_sweep*Tf, perc_silence*Tf, SR, 20, SR/2, 0.95, 1e-2, false);
    end
else
    error("Input should be set to impulse, sweep or file!");
end

% derive grid spacing from stability condition [1,2]
htest = (heps:heps:0.1).';
ineq = (htest > sqrt(k*K*(2*eta + sqrt(4*eta^2 + (1 + abs(cos(q*htest))).^2))));
idx = find(ineq == 1, 1);
hmin = max([2*gamma*k*sqrt(theta), htest(idx)]);
N = floor(1/hmin);
h = 1/N;
q_ = (2/h)*sin(q*h/2);

% calculate derivative matrices
e = ones(N-1,1);
Dxp = spdiags([-e e], 0:1, N-1,N-1)./h;
Dxm = spdiags([-e e], -1:0, N-1,N-1)./h;
Dxx = spdiags([e -2*e e], -1:1, N-1,N-1)./h^2;
Dxxxx = spdiags([e -4*e 6*e -4*e e], -2:2, N-1,N-1);
Dxxxx(1,1) = 5;
Dxxxx(N-1,N-1) = 5;
Dxxxx = Dxxxx./h^4;

% calculate equation matrices
A1 = (1/k^2)*(speye(N-1) + eta*K*k*Dxx) + gamma^2*q^2*0.5*(1-alpha)*speye(N-1) + sigma_t/k*speye(N-1);
B1 = (-2/k^2)*(speye(N-1) + eta*K*k*Dxx) + K^2*(Dxxxx + 2*q_^2*Dxx + q_^4*speye(N-1)) + gamma^2*q^2*alpha*speye(N-1);
C1 = (1/k^2)*(speye(N-1) + eta*K*k*Dxx) + gamma^2*q^2*0.5*(1-alpha)*speye(N-1) - sigma_t/k*speye(N-1);
D1 = -gamma^2*q^2*0.5*(1-alpha)*Dxm;
E1 = -gamma^2*q^2*alpha*Dxm;
F1 = D1;
A2 = gamma^2*0.5*(1-alpha)*Dxp;
B2 = gamma^2*alpha*Dxp;
C2 = A2;
D2 = (1/k^2)*(speye(N-1) + theta*gamma^2*k^2*Dxx) - gamma^2*0.5*(1-alpha)*Dxx + sigma_l/k*speye(N-1);
E2 = (-2/k^2)*(speye(N-1) + theta*gamma^2*k^2*Dxx) - gamma^2*alpha*Dxx;
F2 = (1/k^2)*(speye(N-1) + theta*gamma^2*k^2*Dxx) - gamma^2*0.5*(1-alpha)*Dxx - sigma_l/k*speye(N-1);
A = [A1, D1; A2, D2];
B = -[B1, E1; B2, E2];
C = -[C1, F1; C2, F2];

% calculate force spreading function
Jt = sparse(N-1,1);
Jl = sparse(N-1,1);
Jt(1) = sin(th_exc)*q;
Jl(1) = cos(th_exc);
J = [Jt; Jl];

% main loop
u = zeros(N-1,1);
u1 = zeros(N-1,1);
u2 = zeros(N-1,1);
zeta = zeros(N-1,1);
zeta1 = zeros(N-1,1);
zeta2 = zeros(N-1,1);
w = [u;zeta];
w1 = [u1;zeta1];
w2 = [u2;zeta2];
out = zeros(Nf,1);
in = in./max(abs(in));
for n = 1:Nf
    % update
    w = A\(B*w1 + C*w2 + J*in(n));
    % output
    u = w(1:N-1);
    zeta = w(N:end);
    if strcmp(out_type, "force") == 1
        Ft = -K^2*(Dxxxx*u1 + 2*q_^2*Dxx*u1 + q_^4*u1) + gamma^2*q^2*alpha*(Dxm*zeta1-u1) + gamma^2*q^2*0.5*(1-alpha)*(Dxm*zeta-u+Dxm*zeta2-u2);
        Fl = gamma^2*alpha*(Dxx*zeta1-Dxp*u1) + gamma^2*0.5*(1-alpha)*(Dxx*zeta-Dxp*u+Dxx*zeta2-Dxp*u2);
        out(n) = -(sin(th_pick)/q)*Ft(end) - cos(th_pick)*Fl(end);
    else
        out(n) = sin(th_pick)*u(end) + cos(th_pick)*zeta(end);
    end
    % shift
    w2 = w1;
    w1 = w;
    u2 = u1;
    u1 = u;
    zeta2 = zeta1;
    zeta1 = zeta;
end

% process output
if strcmp(out_type, "velocity") == 1
    out = diff(out);
end
if strcmp(input, "impulse") == 1
    % listen, plot spectogram and save
    ir = out;
    fig_spec = myspec(ir, SR, 512, 0.75);
    ylim([0 6000]);
    soundsc(ir, SR);
    audiowrite('spring_reverb_ir.wav', ir./max(abs(ir)), SR);
elseif strcmp(input, "sweep") == 1
    % listen, plot spectogram and save
    ir = conv(out, in_f);
    ir(1:length(in_f)) = [];
    fig_spec = myspec(ir, SR, 512, 0.75);
    ylim([0 6000]);
    soundsc(ir, SR);
    audiowrite('spring_reverb_ir.wav', ir./max(abs(ir)), SR);
elseif strcmp(input, "file") == 1
    % listen and save
    out = out./max(abs(out));
    out_drywet = (1 - rev_drywet)*in + rev_drywet*out;
    soundsc(out_drywet, SR);
    [~,file_name] = fileparts(file_path);
    audiowrite(append(file_name, '_spring_reverb', '.wav'), out_drywet, SR);
end

%% FUNCTIONS
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