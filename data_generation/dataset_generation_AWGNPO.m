root = 'C:/Users/amc/matlab_awgn_8class_1kpts/';

%% Parameters
len_patch = 2000;       % Length of patch
Nsym = 10;              % Filter span in symbol durations
beta = 0.25;            % Roll-off factor
sampsPerSym = 2;        % Upsampling factor

rcrFilt = comm.RaisedCosineReceiveFilter(...
    "Shape",                "Square root", ...
    "RolloffFactor",        beta, ...
    "FilterSpanInSymbols",  Nsym, ...
    "InputSamplesPerSymbol", sampsPerSym, ...
    'DecimationFactor',     1, ...
    'Gain',                 1);

modulationTypesStr = cellstr([ ...
    'BPSK  '; 'QPSK  ';
    '8PSK  '; '16QAM ';
    '32QAM '; '64QAM ';
    '128QAM'; '256QAM';]);
modulationTypes = [2 4 8 16 32 64 128 256];
numModulationTypes = length(modulationTypes);

noiseSnrList = 10:2:28;
lenNoiseSnrList = length(noiseSnrList);

%% Data Generation

for M = modulationTypes
    if M == 2
        phaseOffsetList = 0:5:175;
        numPatch = 8;
    elseif M == 8
        phaseOffsetList = 0:5:40;
        numPatch = 32;
    else
        phaseOffsetList = 0:5:85;
        numPatch = 16;
    end
    numInstance = length(phaseOffsetList)*numPatch;

    for noiseSnr = noiseSnrList
        dataset = zeros(numInstance, 1000);
        cnt = 0;

        for patch = 1:numPatch
            for phaseOffset = phaseOffsetList
                cnt = cnt + 1;

                % Generate random data
                x = randi([0 M-1],len_patch,1);

                % Modulate
                filterCoeffs = rcosdesign(beta, Nsym, sampsPerSym);
                if (M == 2) || (M == 8)
                    syms = pskmod(x,M);
                elseif M ==4
                    syms = pskmod(x,4,pi/4);
                else
                    syms = qammod(x,M);
                    meanPower = mean(abs(syms).^2);
                    syms = syms/sqrt(meanPower);
                end
                y = filter(filterCoeffs, 1, upsample(syms,sampsPerSym));

                % AWGN channel -> Phase Offset
                yc = awgn(y, noiseSnr);
                yc = yc * exp(1i*2*pi*phaseOffset/360);

                % Normalize average power to 1
                avg_pow = sum(abs(yc).^2)/length(yc);
                yc = yc / sqrt(avg_pow);

                % MF
                yr = step(rcrFilt, yc);

                % Decimation
                yr_ = yr(1:sampsPerSym:end);

                % Crop to desired length
                yf = yr_(1001:end);

                % Normalize average power to 1
                avg_pow = sum(abs(yf).^2)/length(yf);
                yf = yf / sqrt(avg_pow);

                % Visualization
                if patch==1 && phaseOffset==0 && noiseSnr==28
                    scatterplot(yf)
                end

                % Store in array
                dataset(cnt,:) = yf;
            end
        end

        % Save data
        save(sprintf('%s%s_%sdB.mat',root,char(modulationTypesStr(log2(M))),int2str(noiseSnr)), 'dataset')

    end
end





