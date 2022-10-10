function [y1, y2] = helperModClassFrameGenerator(x1, x2, windowLength, stepSize, offset)

numSamples = length(x1);
numFrames = ...
  floor(((numSamples-offset)-(windowLength-stepSize))/stepSize);

y1 = zeros([windowLength,numFrames],class(x1));
y2 = zeros([windowLength,numFrames],class(x2));

startIdx = 1;
frameCnt = 1;

while startIdx + windowLength < numSamples
  xWindowed = x1(startIdx+(0:windowLength-1),1);
  framePower = mean(abs(xWindowed).^2);
  xWindowed = xWindowed / sqrt(framePower);
  y1(:,frameCnt) = xWindowed;

  xWindowed = x2(startIdx+(0:windowLength-1),1);
  framePower = mean(abs(xWindowed).^2);
  xWindowed = xWindowed / sqrt(framePower);
  y2(:,frameCnt) = xWindowed;

  frameCnt = frameCnt + 1;
  startIdx = startIdx + stepSize;
end