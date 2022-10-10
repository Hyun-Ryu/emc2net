classdef helperModClassTestChannel < matlab.System
  properties
    SNR = 20
    CenterFrequency = 2.4e9
  end

  properties (Nontunable)
    SampleRate = 1
    PathDelays = 0
    AveragePathGains = 0
    KFactor = 3
    MaximumDopplerShift = 0
    MaximumClockOffset = 0
  end

  properties(Access = private)
    MultipathChannel
    FrequencyShifter
    TimingShifter
    C % 1+(ppm/1e6)
  end

  methods
    function obj = helperModClassTestChannel(varargin)
      % Support name-value pair arguments when constructing object
      setProperties(obj,nargin,varargin{:})
    end
  end
  
  methods(Access = protected)
    function setupImpl(obj)
      obj.MultipathChannel = comm.RicianChannel(...
        'SampleRate', obj.SampleRate, ...
        'PathDelays', obj.PathDelays, ...
        'AveragePathGains', obj.AveragePathGains, ...
        'KFactor', obj.KFactor, ...
        'MaximumDopplerShift', obj.MaximumDopplerShift);
    end

    function y = stepImpl(obj,x)
      % Add channel impairments
      yInt1 = addMultipathFading(obj,x);
      y     = addNoise(obj, yInt1);
    end

    function out = addMultipathFading(obj, in)     
      % Get new path gains
      reset(obj.MultipathChannel)
      % Pass input through the new channel
      out = step(obj.MultipathChannel, in);
    end
    
    function out = addNoise(obj, in)
      out = awgn(in,obj.SNR);
    end
    
    function resetImpl(obj)
      reset(obj.MultipathChannel);
      reset(obj.FrequencyShifter);
    end

    function s = infoImpl(obj)
      if isempty(obj.MultipathChannel)
        setupImpl(obj);
      end
      
      % Get channel delay from fading channel object delay
      mpInfo = info(obj.MultipathChannel);
      
      % Calculate maximum frequency offset
      maxClockOffset = obj.MaximumClockOffset;
      maxFreqOffset = (maxClockOffset / 1e6) * obj.CenterFrequency;
      
      % Calculate maximum timing offset
      maxClockOffset = obj.MaximumClockOffset;
      maxSampleRateOffset = (maxClockOffset / 1e6) * obj.SampleRate;
      
      s = struct('ChannelDelay', ...
        mpInfo.ChannelFilterDelay, ...
        'MaximumFrequencyOffset', maxFreqOffset, ...
        'MaximumSampleRateOffset', maxSampleRateOffset);
    end
  end
end