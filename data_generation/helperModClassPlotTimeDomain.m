function helperModClassPlotTimeDomain(dataDirectory,modulationTypes,fs)

numRows = ceil(length(modulationTypes) / 4);
for modType=1:length(modulationTypes)
  subplot(numRows, 4, modType);
  files = dir(fullfile(dataDirectory,"*" + string(modulationTypes(modType)) + "*"));
  idx = randi([1 length(files)]);
  load(fullfile(files(idx).folder, files(idx).name), 'frame_input');

  frame_input = frame_input(1:500);
  spf = size(frame_input,1);
  t = 1000*(0:spf-1)/fs;

  plot(t,real(frame_input), '-'); grid on; axis equal; axis square
  hold on
  plot(t,imag(frame_input), '-'); grid on; axis equal; axis square
  hold off
  title(string(modulationTypes(modType)));
  xlabel('Time (ms)'); ylabel('Amplitude')
end
end