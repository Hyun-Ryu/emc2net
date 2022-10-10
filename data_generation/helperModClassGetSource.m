function src = helperModClassGetSource(modType, sps, spf)
switch modType
  case {"BPSK"}
    M = 2;
  case {"QPSK"}
    M = 4;
  case "8PSK"
    M = 8;
  case "16QAM"
    M = 16;
  case "32QAM"
    M = 32;
  case "64QAM"
    M = 64;
  case "128QAM"
    M = 128;
  case "256QAM"
    M = 256;
end
  src = @()randi([0 M-1],spf/sps,1);
end