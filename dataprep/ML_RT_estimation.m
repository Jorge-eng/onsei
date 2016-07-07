function [rt_est,rt_est_mean, RT_est, rte_handle ] = ML_RT_estimation(x,fs,block_size,overlap,remove_from_avg)
%--------------------------------------------------------------------------
% Check input parameters
%--------------------------------------------------------------------------
[num_ch,num_samples] = size(x);

if nargin < 2
    error('not enough input parameters');
end

if (num_ch ~= 1)
    error('Input signals must be single channel vectors [1xN]');
end

if ~exist('fs','var')
    fs = 16000;
end
if ~exist('block_size','var')
    block_size = round(20e-3*fs);
end
if ~exist('overlap','var')
    overlap = round(block_size/2);
end
if ~exist('remove_from_avg','var')
    remove_from_avg = [10,1];
end

%--------------------------------------------------------------------------
% Get parameters
%--------------------------------------------------------------------------

if num_samples/fs < remove_from_avg(1)
    error('remove_from_avg(1) larger than given speech signal');
end


%--------------------------------------------------------------------------
% Initialize RT estimation
%--------------------------------------------------------------------------
rte_handle = ML_RT_estimation_init(fs);

%--------------------------------------------------------------------------
% Block processing
%--------------------------------------------------------------------------

% Note
% Block size and overlap are not indentical to the block sizes and block
% shifts given by rte_handle to demonstrate the integration of the RT
% estimation into a processing scheme with a given block size and frame shift
% (including sample wise processing as special case).

% vectors for all RT estimates
rt_est = ones( ceil(length(x)/rte_handle.N_shift)+1, 1)*rte_handle.RT_initial;
RT_est = zeros( ceil(length(x)/overlap)+1, 1);

% initialize counters
k = 0;
rt_frame_cnt = 0;
RT = rte_handle.RT_initial;
k_rt = rte_handle.N*rte_handle.down +1;    % index counter for RT estimation

for cnt = 1:overlap:num_samples-block_size+1
    k = k+1;  % frame counter for overall block processing scheme
    
    %----------------------------------------------------------------------
    % New T60 Estimation
    %----------------------------------------------------------------------
    if cnt > k_rt
        rt_frame_cnt = rt_frame_cnt + 1; % frame counter for RT estimation
        
        x_seg = x( k_rt - rte_handle.N*rte_handle.down + 1 : rte_handle.down : k_rt );
        [RT, rte_handle] =  ML_RT_estimation_frame( x_seg, rte_handle ); % actual RT estimation
        k_rt = k_rt + rte_handle.N_shift;  % increase index counter for RT estimation
        
        rt_est(rt_frame_cnt) = RT;% save RT estimate over time
    end
    %----------------------------------------------------------------------
    
    RT_est( k ) = RT;
end

RT_est = RT_est(1:k);
rt_est = rt_est(1:rt_frame_cnt);

%--------------------------------------------------------------------------
% Mean RT, averaged over all considered frames
%--------------------------------------------------------------------------
fr2sec_idx = linspace(1,num_samples/fs,rt_frame_cnt);
idx_tmp = find(fr2sec_idx > remove_from_avg(1));
if isempty(idx_tmp)
    error('input signal is too short for given frame removal');
end
idx(1) = idx_tmp(1);
idx_tmp = find(fr2sec_idx < (fr2sec_idx(end)-remove_from_avg(2)));
idx(2) = idx_tmp(end);
avg_range = idx(1):idx(2);

rt_est_mean = mean(rt_est(avg_range));

function [ RT, par ] = ML_RT_estimation_frame( frame, par )

if length(frame) < par.N_sub
    error('input frame too short')
end

[ M , N ] = size( frame );
if M>N
    h = frame.';
else
    h = frame;
end
% ensures a column vector


cnt = 0;     % sub-frame counter for pre-selection of possible sound decay
RTml = -1;   % default RT estimate (-1 indicates no new RT estimate)

% calculate variance, minimum and maximum of first sub-frame
seg = frame( 1 : par.N_sub );

var_pre = var( seg );
min_pre = min( seg );
max_pre = max( seg );

for k = 2 : par.nos_max,
    
    % calculate variance, minimum and maximum of succeding sub-frame
    seg = frame( 1+(k-1)*par.N_sub : k*par.N_sub );
    var_cur = var( seg );
    max_cur = max( seg );
    min_cur = min( seg );
    
    % -- Pre-Selection of suitable speech decays --------------------
    
    if (var_pre > var_cur) && (max_pre > max_cur) && (min_pre < min_cur)
        % if variance, maximum decraease and minimum increase
        % => possible sound decay detected
        
        cnt = cnt + 1;
        
        % current values becomes previous values
        var_pre = var_cur;
        max_pre = max_cur;
        min_pre = min_cur;
        
    else
        
        if cnt >= par.nos_min % minimum length for assumed sound decay achieved?
            
            % -- Maximum Likelihood (ML) Estimation of the RT
            RTml = max_loglf( frame(1:cnt*par.N_sub), par.a, par.Tquant);
            
        end
        
        break
        
    end
    
    if k == par.nos_max % maximum frame length achieved?
        
        RTml = max_loglf( frame(1:cnt*par.N_sub), par.a, par.Tquant );
        
    end
    
end % eof sub-frame loop


if RTml >= 0  % new ML estimate calculated
    
    % apply order statistics to reduce outliers
    par.hist_counter = par.hist_counter+1;
    
    for i = 1: par.no_bins,
        
        % find index corresponding to the ML estimate
        if  ( RTml >= par.hist_limits(i) ) && ( RTml <= par.hist_limits(i+1) )
            
            index = i;
            break
        end
    end
    
    % update histogram with ML estimates for the RT
    par.hist_rt( index ) = par.hist_rt( index ) + 1;
    
    if par.hist_counter > par.buffer_size +1
        % remove old values from histogram
        par.hist_rt( par.buffer( 1 ) ) = par.hist_rt( par.buffer( 1 ) ) - 1;
    end
    
    par.buffer = [ par.buffer(2:end), index ]; % update buffer with indices
    [ dummy, idx ] = max( par.hist_rt );       % find index for maximum of the histogram
    
    par.RT_raw = par.Tquant( idx );   % map index to RT value
    
end

% final RT estimate obtained by recursive smoothing
RT = par.alpha * par.RT_last + (1-par.alpha) * par.RT_raw;
par.RT_last = RT;

par.RTml = RTml;    % intermediate ML estimate for later analysis


return

%--------------------------------------------------------------------------
function [ ML, ll ] = max_loglf(h, a, Tquant)
%--------------------------------------------------------------------------
% [ ML, ll ] = max_loglf( h, a, Tquant )
%
% returns the maximum of the log-likelihood (LL) function and the LL
% function itself for a finite set of decay rates
%
% Input arguments
% h: input frame
% a: finite set of values for which the max. should be found
% T: corresponding RT values for vector a
%
% Output arguments
% ML : ML estimate for the RT
% ll : underlying LL-function


N = length(h);
n = (0:N-1); % indices for input vector
ll = zeros(length(a),1);

h_square = (h.^2).';

for i=1:length(a),
    
    Sum  = ( a(i).^(-2*n) ) * h_square ;
    
    if Sum < 1e-12
        ll( i ) = -inf;
    else
        ll( i ) = -N/2*( (N-1)*log( a(i) ) + log( 2*pi/N * Sum ) + 1 );
    end
    
end

[ dummy, idx ] = max( ll ); % maximum of the log-likelihood function
ML = Tquant( idx );         % corresponding ML estimate for the RT


return
%--------------------------------------------------------------------------

function rte_handle = ML_RT_estimation_init(fs)

% general parameters
rte_handle.fs = fs;          % sampling frequency
no = rte_handle.fs / 24e3 ;  % correction factor to account for different sampling frequency

if fs<8e3 || fs>24e3
    warning('Algorithm has not been tested for this sampling frequency!')
end

% pararmeters for pre-selection of suitable segments
if fs>8e3
    rte_handle.down = 2;                               % rate for downsampling before RT estimation to reduce computational complexity
else
    rte_handle.down = 1;
end
rte_handle.N_sub = round( no * 820/rte_handle.down);   % sub-frame length (after downsampling)
rte_handle.N_shift = round(rte_handle.N_sub*rte_handle.down/4); % frame shift (before downsampling)
rte_handle.nos_min = 3;                                % minimal number of subframes to detect a sound decay
rte_handle.nos_max = 7;                                % maximal number of subframes to detect a sound decay
rte_handle.N = rte_handle.nos_max*rte_handle.N_sub;    % maximal frame length (after downsampling)

% parameters for ML-estimation
Tmax = 1.2;                  % max RT being considered
Tmin = 0.2;                  % min RT being considered
rte_handle.bin = 0.1;                                  % step-size for RT estimation
rte_handle.Tquant = ( Tmin : rte_handle.bin : Tmax );  % set of qunatized RTs considered for maximum search
rte_handle.a = exp( - 3*log(10) ./ ( (rte_handle.Tquant) .* (rte_handle.fs/rte_handle.down)));   % corresponding decay rate factors
rte_handle.La = length( rte_handle.a );                % no. of considered decay rate factors (= no of. RTs)

% paramters for histogram-based approach to reduce outliers (order statistics)
rte_handle.buffer_size = round( no*1200/rte_handle.down); % buffer size
rte_handle.buffer = zeros( 1, rte_handle.buffer_size );  % buffer with previous indices to update histogram
rte_handle.no_bins  = rte_handle.La;                     % no. of histogram bins
rte_handle.hist_limits = Tmin - rte_handle.bin/2 : rte_handle.bin :  Tmax + rte_handle.bin/2 ; % limits of histogram bins
rte_handle.hist_rt = zeros(1,rte_handle.no_bins);        % histogram with ML estimates
rte_handle.hist_counter = 0;                             % counter increased if histogram is updated

% paramters for recursive smoothing of final RT estimate
rte_handle.alpha = 0.996;          % smoothing factor
rte_handle.RT_initial = 0.3;       % initial RT estimate
rte_handle.RT_last = rte_handle.RT_initial; % last RT estimate
rte_handle.RT_raw  = rte_handle.RT_initial; % raw RT estimate obtained by histogram-approach

%--------------------------------------------------------------------------
