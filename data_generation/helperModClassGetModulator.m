function modulator = helperModClassGetModulator(modType, sps)

switch modType
  case "BPSK"
    modulator = @(x)bpskModulator(x,sps);
  case "QPSK"
    modulator = @(x)qpskModulator(x,sps);
  case "8PSK"
    modulator = @(x)psk8Modulator(x,sps);
  case "16QAM"
    modulator = @(x)qam16Modulator(x,sps);
  case "32QAM"
    modulator = @(x)qam32Modulator(x,sps);
  case "64QAM"
    modulator = @(x)qam64Modulator(x,sps);
  case "128QAM"
    modulator = @(x)qam128Modulator(x,sps);
  case "256QAM"
    modulator = @(x)qam256Modulator(x,sps);
end
end

function y = bpskModulator(x,sps)
persistent filterCoeffs
if isempty(filterCoeffs)
  filterCoeffs = rcosdesign(0.35, 4, sps);
end
% Modulate
syms = pskmod(x,2);
% Pulse shape
y = filter(filterCoeffs, 1, upsample(syms,sps));
end

function y = qpskModulator(x,sps)
persistent filterCoeffs
if isempty(filterCoeffs)
  filterCoeffs = rcosdesign(0.35, 4, sps);
end
% Modulate
syms = pskmod(x,4,pi/4);
% Pulse shape
y = filter(filterCoeffs, 1, upsample(syms,sps));
end

function y = psk8Modulator(x,sps)
persistent filterCoeffs
if isempty(filterCoeffs)
  filterCoeffs = rcosdesign(0.35, 4, sps);
end
% Modulate
syms = pskmod(x,8);
% Pulse shape
y = filter(filterCoeffs, 1, upsample(syms,sps));
end

function y = qam16Modulator(x,sps)
persistent filterCoeffs
if isempty(filterCoeffs)
  filterCoeffs = rcosdesign(0.35, 4, sps);
end
% Modulate and Normalize to unit avg power
syms = qammod(x,16);
meanPower = mean(abs(syms).^2);
syms = syms/sqrt(meanPower);
% Pulse shape
y = filter(filterCoeffs, 1, upsample(syms,sps));
end

function y = qam32Modulator(x,sps)
persistent filterCoeffs
if isempty(filterCoeffs)
  filterCoeffs = rcosdesign(0.35, 4, sps);
end
% Modulate and Normalize to unit avg power
syms = qammod(x,32);
meanPower = mean(abs(syms).^2);
syms = syms/sqrt(meanPower);
% Pulse shape
y = filter(filterCoeffs, 1, upsample(syms,sps));
end

function y = qam64Modulator(x,sps)
persistent filterCoeffs
if isempty(filterCoeffs)
  filterCoeffs = rcosdesign(0.35, 4, sps);
end
% Modulate and Normalize to unit avg power
syms = qammod(x,64);
meanPower = mean(abs(syms).^2);
syms = syms/sqrt(meanPower);
% Pulse shape
y = filter(filterCoeffs, 1, upsample(syms,sps));
end

function y = qam128Modulator(x,sps)
persistent filterCoeffs
if isempty(filterCoeffs)
  filterCoeffs = rcosdesign(0.35, 4, sps);
end
% Modulate and Normalize to unit avg power
syms = qammod(x,128);
meanPower = mean(abs(syms).^2);
syms = syms/sqrt(meanPower);
% Pulse shape
y = filter(filterCoeffs, 1, upsample(syms,sps));
end

function y = qam256Modulator(x,sps)
persistent filterCoeffs
if isempty(filterCoeffs)
  filterCoeffs = rcosdesign(0.35, 4, sps);
end
% Modulate and Normalize to unit avg power
syms = qammod(x,256);
meanPower = mean(abs(syms).^2);
syms = syms/sqrt(meanPower);
% Pulse shape
y = filter(filterCoeffs, 1, upsample(syms,sps));
end