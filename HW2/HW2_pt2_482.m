%% Piano
[y,Fs] = audioread('music1.wav');

L = length(y)/Fs;  % record time in seconds
t = (1:length(y))/Fs;
n = length(y);
k = (2*pi/L)*[0:n/2-1 -n/2:-1]; ks = fftshift(k);
tslide_p=0:0.1:L;

v = y'; % transpose of audio
vt = abs(fft(v));
vt_norm = fftshift(vt/max(vt));

figure(1)
subplot(2,1,1)
plot(t,y);
xlabel('Time [sec]'); 
ylabel('Amplitude');
title('Mary had a little lamb (piano)');
%p8 = audioplayer(y,Fs); playblocking(p8);

% Fourier domain
subplot(2,1,2) 
plot(ks, vt_norm);
xlabel('frequency (\omega)'), ylabel('FFT(v)')

tau = 10; % some time
a = 25; % larger a is narrower
g = exp(-a*(t-tau).^2);
subplot(2,1,1)
plot(t,v,'k',t,g,'m','Linewidth',2) 
set(gca,'Fontsize',16), xlabel('Time (t)'), ylabel('v(t)')

%% Piano spectrogram

piano_notes = [];
vgt_spec = zeros(length(tslide_p),n);

% figure(3)
for j=1:length(tslide_p)
    g = exp(-a*(t-tslide_p(j)).^2);
    vg = g.*v;
    vgt=abs(fft(vg));
    [M, I] = max(vgt);
    piano_notes = [piano_notes; abs(k(I))/(2*pi)]; 
    vgt_spec(j,:) = fftshift(vgt);
%     subplot(2,1,1), plot(t, vg, 'k')
%     subplot(2,1,2), plot(ks,fftshift(vgt/max(vgt)))
%     drawnow
%     pause(0.1)
end

%% Plot spectrogram
% min(piano_notes) around 200
% max(piano notes) around 350
figure(4)
pcolor(tslide_p,ks./(2*pi),vgt_spec.'), shading interp
xlabel("Time (s)"); ylabel("Frequency (Hz)"); title("Piano Spectrogram");
set(gca,'Ylim',[200 400])
colormap(hot)

%% Recorder
[y,Fs] = audioread('music2.wav');

L = length(y)/Fs;  % record time in seconds
t = (1:length(y))/Fs;
n = length(y);
k = (2*pi/L)*[0:n/2-1 -n/2:-1]; ks = fftshift(k);
tslide_r=0:0.1:L;

v = y'; % transpose of audio
vt = abs(fft(v));
vt_norm = fftshift(vt/max(vt));

figure(1)
subplot(2,1,1)
plot(t,y);
xlabel('Time [sec]'); 
ylabel('Amplitude');
title('Mary had a little lamb (recorder)');
% p8 = audioplayer(y,Fs); playblocking(p8);

% Fourier domain
subplot(2,1,2) 
plot(ks, vt_norm);
xlabel('frequency (\omega)'), ylabel('FFT(v)')

tau = 10; % some time
a = 25; % larger a is narrower
g = exp(-a*(t-tau).^2);
subplot(2,1,1)
plot(t,v,'k',t,g,'m','Linewidth',2) 
set(gca,'Fontsize',16), xlabel('Time (t)'), ylabel('v(t)')

%% Recorder spectrogram

recorder_notes = [];
vgt_spec = zeros(length(tslide_r),n);

% figure(3)
for j=1:length(tslide_r)
    g = exp(-a*(t-tslide_r(j)).^2);
    vg = g.*v;
    vgt=abs(fft(vg));
    [M, I] = max(vgt);
    recorder_notes = [recorder_notes; abs(k(I))/(2*pi)]; 
    vgt_spec(j,:) = fftshift(vgt);
%     subplot(2,1,1), plot(t, vg, 'k')
%     subplot(2,1,2), plot(ks,fftshift(vgt/max(vgt)))
%     drawnow
%     pause(0.1)
end
%% Plot recorder spectrogram
figure(5)
pcolor(tslide_r,ks./(2*pi),vgt_spec.'), shading interp
xlabel("Time (s)"); ylabel("Frequency (Hz)"); title("Recorder Spectrogram");
set(gca,'Ylim',[600 1200])
colormap(hot)

%% Plot notes
figure(6)
plot(tslide_p, piano_notes, 'o');
yticks([261.63,293.66,311.13,329.63,349.23, 369.99]);
yticklabels({'Middle C','C#','D','D#','E','F', 'F#'});
title("Piano score (250-350 Hz)");

figure(7)
plot(tslide_r, recorder_notes, 'o');
yticks([783.99,830.61,880,932.33,987.77,1046.5,1108.7]);
yticklabels({'G','G#','A','A#','B','C','C#'});
title("Recorder score (800-1100 Hz)");

