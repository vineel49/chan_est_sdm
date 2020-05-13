% Channel estimation using steepest descent method

close all
clear all
clc
%---------------- SIMULATION PARAMETERS ------------------------------------
SNR_dB = 40; % SNR PER BIT (dB)
NUM_FRAMES = 1*(10^3); % SIMULATION RUNS
FFT_LEN = 1024; % LENGTH OF THE FFT/IFFT
CHAN_LEN = 10; % NUMBER OF CHANNEL TAPS
FADE_VAR_1D = 0.5; % 1D VARIANCE OF THE FADE CHANNEL
PREAMBLE_LEN = 512; % LENGTH OF THE PREAMBLE
CP_LEN = CHAN_LEN-1; % LENGTH OF THE CYCLIC PREFIX
NUM_BIT = 2*FFT_LEN; % NUMBER OF DATA BITS (OVERALL RATE IS 2)
EST_CHAN_LEN = CHAN_LEN; % LENGTH OF THE EQUALIZER
ITERATIONS = 100; % ITERATIONS FOR THE SDM

% SNR PARAMETERS
SNR = 10^(0.1*SNR_dB); % LINEAR SCALE
NOISE_VAR_1D = 0.5*2*2*FADE_VAR_1D*CHAN_LEN/(2*FFT_LEN*SNR); % 1D VARIANCE OF AWGN
%--------------------------------------------------------------------------
%                        PREAMBLE GENERATION
PREAMBLE_A = randi([0 1],1,2*PREAMBLE_LEN);
PREAMBLE_QPSK = 1-2*PREAMBLE_A(1:2:end)+1i*(1-2*PREAMBLE_A(2:2:end));
% AVERAGE POEWR OF PREAMBLE PART MUST BE EQUAL TO DATA PART
PREAMBLE_QPSK = sqrt(PREAMBLE_LEN/FFT_LEN)*PREAMBLE_QPSK; % (4) IN PAPER
PREAMBLE_QPSK_IFFT = ifft(PREAMBLE_QPSK);
%--------------------------------------------------------------------------
C_MSE = 0; % SQUARED ERROR OF CIR AFTER SDM
C_BER = 0; % BIT ERRORS IN EACH FRAME
tic()
%--------------------------------------------------------------------------
AUTO_CORR_VEC = xcorr(PREAMBLE_QPSK_IFFT,PREAMBLE_QPSK_IFFT,'unbiased');
MID_POINT = (length(AUTO_CORR_VEC)+1)/2;
C = AUTO_CORR_VEC(MID_POINT:MID_POINT+EST_CHAN_LEN-1); % FIRST COLUMN OF TOEPLITZ MATRIX
R = fliplr(AUTO_CORR_VEC(MID_POINT-EST_CHAN_LEN+1:MID_POINT)); % FIRST ROW OF TOEPLITZ MATRIX
Rvv_MATRIX = toeplitz(C,R);
%-------------------------------
MAX_STEP_SIZE = 1/real(max(eig(Rvv_MATRIX))); % MAXIMUM STEP SIZE
STEP_SIZE = 0.125*MAX_STEP_SIZE;

%--------------------------------------------------------------
for FRAME_CNT = 1:NUM_FRAMES
%                           TRANSMITTER
% SOURCE
A = randi([0 1],1,NUM_BIT); 

% QPSK mapping 
MOD_SIG = 1-2*A(1:2:end) + 1i*(1-2*A(2:2:end));

% IFFT OPERATION
T_QPSK_SIG = ifft(MOD_SIG); 

% INSERTING CYCLIC PREFIX AND PREAMBLE
T_TRANS_SIG = [PREAMBLE_QPSK_IFFT T_QPSK_SIG(end-CP_LEN+1:end) T_QPSK_SIG]; 
%--------------------------------------------------------------------------
%                            CHANNEL   
% RAYLEIGH FADING CHANNEL
FADE_CHAN = sqrt(FADE_VAR_1D)*randn(1,CHAN_LEN) + 1i*sqrt(FADE_VAR_1D)*randn(1,CHAN_LEN);     

% AWG
AWGN = sqrt(NOISE_VAR_1D)*randn(1,FFT_LEN + CP_LEN + PREAMBLE_LEN + CHAN_LEN - 1) ...
    + 1i*sqrt(NOISE_VAR_1D)*randn(1,FFT_LEN + CP_LEN + PREAMBLE_LEN + CHAN_LEN - 1); 

% CHANNEL OUTPUT
CHAN_OP = conv(T_TRANS_SIG,FADE_CHAN) + AWGN; % Chan_Op stands for channel output
%--------------------------------------------------------------------------
%                          RECEIVER 
% CHANNEL ESTIMATION USING SDM
CROSS_CORR_VEC = xcorr(CHAN_OP(1:length(PREAMBLE_QPSK_IFFT)),PREAMBLE_QPSK_IFFT,'unbiased');
MID_POINT = (length(CROSS_CORR_VEC)+1)/2;
IP_CROSS_CORR_VEC = CROSS_CORR_VEC(MID_POINT:MID_POINT+EST_CHAN_LEN-1).';

EST_FADE_CHAN = zeros(EST_CHAN_LEN,1); % INITIALIZATION

for i1 = 1:ITERATIONS
EST_FADE_CHAN = EST_FADE_CHAN + STEP_SIZE*(IP_CROSS_CORR_VEC - Rvv_MATRIX*EST_FADE_CHAN);  
end

EST_FADE_CHAN = EST_FADE_CHAN.'; % NOW A ROW VECTOR
% COMPUTING MEAN SQUARED ERROR OF THE ESTIMATED CHANNEL IMPULSE RESPONSE
ERROR = EST_FADE_CHAN - FADE_CHAN;
C_MSE = C_MSE + ERROR*ERROR';

%-----------------------------------------------------------------------
EST_FREQ_RESP = fft(EST_FADE_CHAN,FFT_LEN);
% discarding preamble
CHAN_OP(1:PREAMBLE_LEN) = [];
% discarding cyclic prefix and transient samples
CHAN_OP(1:CP_LEN) = [];
T_REC_SIG_NO_CP = CHAN_OP(1:FFT_LEN);
% PERFORMING THE FFT
F_REC_SIG_NO_CP = fft(T_REC_SIG_NO_CP);
% ML DETECTION
QPSK_SYM = [1+1i 1-1i -1+1i -1-1i];
QPSK_SYM1 = QPSK_SYM(1)*ones(1,FFT_LEN);
QPSK_SYM2 = QPSK_SYM(2)*ones(1,FFT_LEN);
QPSK_SYM3 = QPSK_SYM(3)*ones(1,FFT_LEN);
QPSK_SYM4 = QPSK_SYM(4)*ones(1,FFT_LEN);
DIST = zeros(4,FFT_LEN);
DIST(1,:)=(abs(F_REC_SIG_NO_CP - EST_FREQ_RESP.*QPSK_SYM1)).^2; 
DIST(2,:)=(abs(F_REC_SIG_NO_CP - EST_FREQ_RESP.*QPSK_SYM2)).^2;
DIST(3,:)=(abs(F_REC_SIG_NO_CP - EST_FREQ_RESP.*QPSK_SYM3)).^2;
DIST(4,:)=(abs(F_REC_SIG_NO_CP - EST_FREQ_RESP.*QPSK_SYM4)).^2; 
% COMPARING EUCLIDEAN DISTANCE
[~,INDICES] = min(DIST,[],1);
% MAPPING INDICES TO QPSK SYMBOLS
DEC_QPSK_MAP_SYM = QPSK_SYM(INDICES);
% DEMAPPING QPSK SYMBOLS TO BITS
DEC_A = zeros(1,NUM_BIT);
DEC_A(1:2:end) = real(DEC_QPSK_MAP_SYM)<0;
DEC_A(2:2:end) = imag(DEC_QPSK_MAP_SYM)<0;
% CALCULATING BIT ERRORS IN EACH FRAME
C_BER = C_BER + nnz(A-DEC_A);
end
toc()
% bit error rate
BER = C_BER/(NUM_BIT*NUM_FRAMES)

% MEAN SQUARE ERROR OF THE CHANNEL ESTIMATION
MSE = C_MSE/(CHAN_LEN*NUM_FRAMES)


