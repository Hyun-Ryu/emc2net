%% Parameters
trainNow = true;
if trainNow
    numFramesPerModType = 2000;
else
    numFramesPerModType = 2;
end

sps = 8;            % Samples per symbol
spf = 8192;         % Samples per frame
symbolsPerFrame = spf / sps;
fs = 200e3;         % Sample rate
SNR = 30;           % signal-to-noise ratio

modulationTypes = cellstr([ ...
    'BPSK  '; 'QPSK  ';'8PSK  '; '16QAM ';
    '32QAM '; '64QAM ';'128QAM'; '256QAM'; ...
    ]);

channel = helperModClassTestChannel(...
    'SampleRate', fs, ...
    'SNR', SNR, ...
    'PathDelays', [0 1.8 3.4] / fs, ...
    'AveragePathGains', [0 -2 -10], ...
    'KFactor', 4, ...
    'MaximumDopplerShift', 4, ...
    'MaximumClockOffset', 5, ...
    'CenterFrequency', 902e6)

% Set the random number generator to a known state to be able to regenerate
% the same frams every time the simulation is run
rng(1235)
tic

numModulationTypes = length(modulationTypes);

channelInfo = info(channel)
transDelay = 50;
dataDirectory = fullfile('C:','Users','amc','data','Rician_30dB_1024sym');
disp(['Data file directory is ', dataDirectory])

fileNameRoot = 'frame';

% Check if data files exist
dataFilesExist = false;
if exist(dataDirectory,'dir')
  files = dir(fullfile(dataDirectory,sprintf("%s*",fileNameRoot)));
  if length(files) == numModulationTypes*numFramesPerModType
    dataFilesExist = true;
  end
end

%% Data Generation

if ~dataFilesExist
  disp("Generating data and saving in data files...")
  [success,msg,msgID] = mkdir(dataDirectory);
  if ~success
    error(msgID,msg)
  end
  for modType = 1:numModulationTypes
    elapsedTime = seconds(toc);
    elapsedTime.Format = 'hh:mm:ss';
    fprintf('%s - Generating %s frames\n', ...
      char(elapsedTime), char(modulationTypes(modType)))
    
    label = char(modulationTypes(modType));
    numSymbols = (numFramesPerModType / sps);
    dataSrc = helperModClassGetSource(label, sps, 2*spf);
    modulator = helperModClassGetModulator(label, sps);

    % Digital modulation types use a center frequency of 902 MHz
    channel.CenterFrequency = 902e6;
    
    for p=1:numFramesPerModType
      % Generate random data
      x = dataSrc();
      
      % Modulate
      y = modulator(x);
      
      % Pass through independent channels
      rxSamples = step(channel, y);
      
      % Remove transients from the beginning, trim to size, and normalize
      [frame_input, frame_gt] = helperModClassFrameGenerator(rxSamples, y, spf, spf, transDelay);
      
      % Save data file
      fileName = fullfile(dataDirectory,...
        sprintf("%s_%s_%04d",fileNameRoot,label,p));
      save(fileName,"frame_input","frame_gt","label")
    end
  end
else
  disp("Data files exist. Skip data generation.")
end

% Plot the amplitude of the real and imaginary parts of the example frames
% against the sample number
helperModClassPlotTimeDomain(dataDirectory,modulationTypes,fs)



