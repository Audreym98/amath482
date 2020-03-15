clear; close all; clc;
load Testdata

L=15; % Spatial domain
n=64; % Fourier modes
x2=linspace(-L,L,n+1); x=x2(1:n); y=x; z=x;
k=(2*pi/(2*L))*[0:(n/2-1) -n/2:-1]; ks=fftshift(k); % Frequency components

[X,Y,Z]=meshgrid(x,y,z);
[Kx,Ky,Kz]=meshgrid(ks,ks,ks);

% Average noisy measurements
Utavg = zeros(n, n, n);
N = 20;
for j=1:N
    Un(:,:,:)=reshape(Undata(j,:),n,n,n);
    % close all, isosurface(X,Y,Z,abs(Un),0.4)
    % axis([-20 20 -20 20 -20 20]), grid on, drawnow
    % pause(1)
    % Apply FFT to get signal in frequency domain
    Unt(:,:,:) = fftn(Un);
    Utavg = Utavg + Unt;
end
% Average signals
Utavg = abs(fftshift(Utavg))/N;
% Make max value 1
Utavg_norm = Utavg/max(Utavg(:));
% Frequency plot after averaging the spectrum
close all, isosurface(Kx, Ky, Kz, Utavg_norm,0.4);
title("Central Frequency After Averaging the Spectrum")
xlabel("Kx")
ylabel("Ky")
zlabel("Kz")
axis([-20 20 -20 20 -20 20]), grid on, drawnow

% Determine the frequency of interest
% Gets the index of the max (target) frequency
[M, I] = max(Utavg_norm(:)); % I is index of max
% Determines target frequency location
[I, J, K] = ind2sub(size(Utavg_norm), I);
% Target frequencies for filter
Kx0 = Kx(I, J, K);
Ky0 = Ky(I, J, K);
Kz0 = Kz(I, J, K);
% Signal filter around target frequency
filter = exp(-0.2 *((Kx - Kx0).^2 + (Ky - Ky0).^2 + (Kz - Kz0).^2));

x_max = zeros(1, N);
y_max = zeros(1, N);
z_max = zeros(1, N);
for j=1:N
     Un(:,:,:)=reshape(Undata(j,:),n,n,n);
     Unt(:,:,:) = fftn(Un);
    Unt_shift(:,:,:) = fftshift(Unt);
    % Apply filter
    Unt_filter = Unt_shift.*filter;
    % Go back to time domain
    Unt_filter = ifftshift(Unt_filter);
    Un_filter = ifftn(Unt_filter);
    % Get marble locations throughout time
    [M, I] = max(Un_filter(:));
    [x_m, y_m, z_m] = ind2sub(size(Un_filter), I);
    x_max(j) = X(x_m, y_m, z_m);
    y_max(j) = Y(x_m, y_m, z_m);
    z_max(j) = Z(x_m, y_m, z_m);
end
Unt_filter_norm = abs(Unt_filter)/max(abs(Unt_filter(:)));
close all, isosurface(Kx, Ky, Kz, fftshift(Unt_filter_norm),0.4);
title("Filtered Signal in the Frequency Domain")
xlabel("Kx")
ylabel("Ky")
zlabel("Kz")
axis([-10 10 -10 10 -10 10]), grid on, drawnow

% Plots path of marble
plot3(x_max, y_max, z_max, 'Color', 'b',...
    'LineWidth', 2); 
grid on
% Adds labels
title('Path of the Marble in the Intestine')
xlabel('x'), 
ylabel('y'),
zlabel('z')
hold on
% Gets marble location at last measurement
marble_end = [x_max(end), y_max(end), z_max(end)];
% Plots marble
plot3(marble_end(1), marble_end(2), marble_end(3), '.', ...
    'Color', 'g', 'markersize', 40)
% Adds marble location to plot
text(marble_end(1), marble_end(2), marble_end(3),...
    ['   (' num2str(marble_end(1)) ', ' num2str(marble_end(2)) ', ' ...
    num2str(marble_end(3)) ')'])
    

