# Transmitter-Receiver

In this project I am sharing some kernels which are very useful in communication system developing.

There are some blocks which are used directly in demodulation and they are necessary in software define radios.

## 1 - DDS

This block transform data from IF or nonbaseband to baseband

## 2 - FilterDownSample


FilterDownSample filter is useful to decimate signals to purpose rate

## 3 - BaseBandFilter

It is necessary to use BaseBandFilter to reduce the noise effect

## 6 - Timing Recovery (symbol synchronizer)

We use this processing block to synchronize **Trasmitter** and **Receiver**

## 9 - Soft-Demapper

By using this kernel we can change symbols of PSK, OQPSK and FSK signals to bit.
