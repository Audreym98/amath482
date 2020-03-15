% Pt 1
%% Plot signal and FFT
clear; close all; clc;
load handel
v = y';
L = length(v)/Fs; % length of audio
t = (1:length(v))/Fs;
n = length(t);
k = (2*pi/L)*[0:(n-1)/2 -(n-1)/2:-1];
ks = fftshift(k);
vt = fft(v);

figure(1)
subplot(2,1,1)
plot(t,v);
xlabel('Time [sec]');
ylabel('Amplitude');
title('Signal of Interest, v(n)');

% Fourier domain
subplot(2,1,2) 
plot(ks, abs(vt)/max(abs(vt)));
xlabel('frequency (\omega)'), ylabel('FFT(v)')

%% plays music
% p8 = audioplayer(v,Fs);
% playblocking(p8);

%% Construct Gabor window and add to time domain plot
tau = 4; % some time
a = 20; % larger a is narrower
g = exp(-a*(t-tau).^2);
subplot(2,1,1)
plot(t,v,'k',t,g,'m','Linewidth',2) 
set(gca,'Fontsize',16), xlabel('Time (t)'), ylabel('v(t)')

%% Apply filter and take fft
vg = g.*v;
vgt = fft(vg);

figure(2)
subplot(3,1,1) 
plot(t,v,'k','Linewidth',2) 
hold on 
plot(t,g,'m','Linewidth',2)
set(gca,'Fontsize',16), xlabel('Time (t)'), ylabel('v(t)')

subplot(3,1,2) 
plot(t,vg,'k','Linewidth',2) 
set(gca,'Fontsize',16), xlabel('Time (t)'), ylabel('vg(t)')

subplot(3,1,3) 
plot(ks,abs(fftshift(vgt))/max(abs(vgt)),'r','Linewidth',2);
set(gca,'Fontsize',16)
xlabel('frequency (\omega)'), ylabel('FFT(vg)')
%% Change window size - narrow window
tau = 4;
a = 80;
g = exp(-a*(t-tau).^2);
vg = g.*v;
vgt = fft(vg);

figure(3)
subplot(3,1,1) 
plot(t,v,'k','Linewidth',2) 
hold on 
plot(t,g,'m','Linewidth',2)
set(gca,'Fontsize',16), xlabel('Time (t)'), ylabel('v(t)')

subplot(3,1,2) 
plot(t,vg,'k','Linewidth',2) 
set(gca,'Fontsize',16), xlabel('Time (t)'), ylabel('vg(t)')

subplot(3,1,3) 
aa = abs(fftshift(vgt))/max(abs(vgt));
plot(ks,aa,'r','Linewidth',2);
set(gca,'Fontsize',16)
xlabel('frequency (\omega)'), ylabel('FFT(vg)')
%% Change window size - wide window
tau = 4;
a = 1;
g = exp(-a*(t-tau).^2);
vg = g.*v;
vgt = fft(vg);

figure(4)
subplot(3,1,1) 
plot(t,v,'k','Linewidth',2) 
hold on 
plot(t,g,'m','Linewidth',2)
set(gca,'Fontsize',16), xlabel('Time (t)'), ylabel('v(t)')

subplot(3,1,2) 
plot(t,vg,'k','Linewidth',2) 
set(gca,'Fontsize',16), xlabel('Time (t)'), ylabel('vg(t)')

subplot(3,1,3) 
aa = abs(fftshift(vgt))/max(abs(vgt));
plot(ks,aa,'r','Linewidth',2);
set(gca,'Fontsize',16)
xlabel('frequency (\omega)'), ylabel('FFT(vg)')
%% Calculate Gabor transform and plot spectrogram
a = 50;
tslide=0:0.1:L;
vgt_spec = zeros(length(tslide),n);
for j=1:length(tslide)
    g=exp(-a*(t-tslide(j)).^2); 
    vg=g.*v; 
    vgt=fft(vg); 
    vgt_spec(j,:) = fftshift(abs(vgt)); % We don't want to scale it
end

figure(5)
pcolor(tslide,ks./(2*pi),vgt_spec.'), 
shading interp 
colormap(hot)
ylabel("Frequency (Hz)");
title('Spectrogram, a = 20')

%% Spectrograms for varying window sizes
% 0.01 looks like FFT, not time information
% 25 better time info, less frequency resolution Heinsberg uncertainity
figure(6)
a_vec = [80 50 20 10 5 1 0.5 0.1];
for jj = 1:length(a_vec)
    a = a_vec(jj);
    tslide=0:0.1:L;
    vgt_spec = zeros(length(tslide),n);
    for j=1:length(tslide)
        g=exp(-a*(t-tslide(j)).^2);
        vg=g.*v;
        vgt=fft(vg);
        vgt_spec(j,:) = fftshift(abs(vgt));
    end
    subplot(3,3,jj)
    pcolor(tslide,ks./(2*pi),vgt_spec.'),
    shading interp
    title(['a = ',num2str(a)],'Fontsize',16)
    colormap(hot) 
end
vgt_spec = repmat(fftshift(abs(vt)),length(tslide),1);
subplot(3,3,9)
pcolor(tslide,ks./(2*pi),vgt_spec.'),
shading interp 
title('fft','Fontsize',16)
colormap(hot)

%% oversampling
a = 10;

figure(7)
subplot(2,1,1)
plot(t,v,'k','Linewidth',2) 
hold on
tslide=0:0.01:L;
for j=1:length(tslide)
    g = exp(-a*(t-tslide(j)).^2);
    subplot(2,1,1)
    plot(t,g,'m','Linewidth',2)
end
set(gca,'Fontsize',16), xlabel('Time (t)'), ylabel('v(t)')

subplot(2,1,2)
vgt_spec = zeros(length(tslide),n);
for j=1:length(tslide)
    g=exp(-a*(t-tslide(j)).^2); 
    vg=g.*v; 
    vgt=fft(vg); 
    vgt_spec(j,:) = fftshift(abs(vgt)); 
end

pcolor(tslide,ks./(2*pi),vgt_spec.'), 
shading interp
title('Oversampling','Fontsize',16)
colormap(hot)
ylabel("Frequency (Hz)");

%% normal sampling
a = 5;

figure(8)
subplot(2,1,1)
plot(t,v,'k','Linewidth',2) 
hold on
tslide=0:0.1:L;
for j=1:length(tslide)
    g = exp(-a*(t-tslide(j)).^2);
    subplot(2,1,1)
    plot(t,g,'m','Linewidth',2)
end
set(gca,'Fontsize',16), xlabel('Time (t)'), ylabel('v(t)')

subplot(2,1,2)
vgt_spec = zeros(length(tslide),n);
for j=1:length(tslide)
    g=exp(-a*(t-tslide(j)).^2); 
    vg=g.*v; 
    vgt=fft(vg); 
    vgt_spec(j,:) = fftshift(abs(vgt)); 
end

pcolor(tslide,ks./(2*pi),vgt_spec.'), 
shading interp
title('Normal','Fontsize',16)
colormap(hot)
ylabel("Frequency (Hz)");

%% undersampling
a = 5;

figure(9)
subplot(2,1,1)
plot(t,v,'k','Linewidth',2) 
hold on
tslide=0:1:L;
for j=1:length(tslide)
    g = exp(-a*(t-tslide(j)).^2);
    subplot(2,1,1)
    plot(t,g,'m','Linewidth',2)
end
set(gca,'Fontsize',16), xlabel('Time (t)'), ylabel('v(t)')

subplot(2,1,2)
vgt_spec = zeros(length(tslide),n);
for j=1:length(tslide)
    g=exp(-a*(t-tslide(j)).^2); 
    vg=g.*v; 
    vgt=fft(vg); 
    vgt_spec(j,:) = fftshift(abs(vgt)); 
end

pcolor(tslide,ks./(2*pi),vgt_spec.'), 
shading interp 
title('Undersampling','Fontsize',16)
colormap(hot)