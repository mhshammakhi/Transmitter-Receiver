clc;clear;

fileID= fopen('pll_out.bin','r');
A=fread(fileID,'float');
fclose(fileID);

signal = A(1:2:end)+1i*A(2:2:end);
clear A;

scatterplot(signal(1000:end))