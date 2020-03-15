%%
clear all; close all; clc; 
%% TEST 1
% add more samples (14 training, 6 test)
alldata = [];
artists = ["ariana_grande", "luke_bryan", "bryce_vine"];
myDir = '/Users/audrey/Desktop/Music/';
songs = 20;
for i = 1:length(artists)
    for j = 1:songs % 5 5" clips from each artist
        [song, Fs] = audioread(strcat(myDir,artists(i),"/",num2str(i),".wav"));
        song = song(:,1); % make mono
        a = song(1:4:22050,:)'; % downsample song
        
        % get spectrogram
         Sgt_spec = [];
         L = 5;
         n = length(a);
         t2 = linspace(0,L,n+1);
         t = t2(1:n);
         tslide = 0:0.1:L;
          for k = 1:length(tslide)
             g = exp(-10*(t - tslide(k)).^2);
             Sg = g.*a;
             Sgt = fft(Sg);
             Sgt_spec = [Sgt_spec; abs(fftshift(Sgt))];
          end
        [m, n] = size(Sgt_spec);
        data = reshape(Sgt_spec, m*n, 1);
        alldata = [alldata data];
    end
end

[u, s, v] = svd(alldata - mean(alldata(:)), 'econ');
%plot(diag(s),'ko','Linewidth',2)
singers = u*v';
% generate random nums to split data
% q = randperm(songs);
% save permutaion
q = [16 15 12 14 13 5 2 6 8 17 18 1 4 20 9 11 7 10 19 3];
%%
features = 20;
tr_size = 14;
test_size = songs - tr_size;

ari = singers(1:features,1:20);
luke = singers(1:features,21:40);
bryce = singers(1:features,41:end);

%% training sets
ari_train = ari(:,q(1:tr_size));
luke_train = luke(:,q(1:tr_size));
bryce_train = bryce(:,q(1:tr_size));

% test sets
ari_test = ari(:,q(tr_size+1:end));
luke_test = luke(:,q(tr_size+1:end));
bryce_test = bryce(:,q(tr_size+1:end));

m1 = mean(ari_train,2);
m2 = mean(luke_train,2);
m3 = mean(bryce_train,2);
m_all = mean(singers);

% Within class
Sw = 0;
for i=1:tr_size
    Sw = Sw + (ari_train(:,i)-m1)*(ari_train(:,i)-m1)';
end
for i=1:tr_size
    Sw = Sw + (luke_train(:,i)-m2)*(luke_train(:,i)-m2)';
end
for i=1:tr_size
    Sw = Sw + (bryce_train(:,i)-m3)*(bryce_train(:,i)-m3)';
end

% Between class
Sb = 0;
Sb = Sb + (m1-m_all)*(m1-m_all)'*tr_size;
Sb = Sb + (m2-m_all)*(m2-m_all)'*tr_size;
Sb = Sb + (m3-m_all)*(m3-m_all)'*tr_size;
%% Project onto single vector
[V2,D] = eig(Sb,Sw);
[~,ind] = max(abs(diag(D)));
w = V2(:,ind); w = w/norm(w,2);

vari = w'*ari_train;
vluke = w'*luke_train;
vbryce = w'*bryce_train;

vari_test = w'*ari_test;
vluke_test = w'*luke_test;
vbryce_test = w'*bryce_test;
pval = [vari_test vluke_test vbryce_test];
answer = [ones(test_size,1); 2*ones(test_size,1); 3*ones(test_size,1)];

disp("means")
disp(mean([vari;vluke;vbryce],2));
disp(sort(mean([vari;vluke;vbryce],2)));
sort_ari = sort(vari);
sort_luke = sort(vluke);
sort_bryce = sort(vbryce);

%% 10 modes: ari (1) < luke < bryce
t1 = tr_size;
t2 = 1;
while sort_luke(t1) > sort_bryce(t2)
    t1 = t1-1;
    t2 = t2+1;
end
thresh_min = (sort_luke(t1)+sort_bryce(t2))/2';

% ari < luke
t1 = tr_size;
t2 = 1;
while sort_ari(t1) > sort_luke(t2)
    t1 = t1-1;
    t2 = t2+1;
end
thresh_max = (sort_ari(t1)+sort_luke(t2))/2';
test = [];
for i=1:test_size*3
    if pval(i) > thresh_max
        test = [test;3];
    elseif pval(i) < thresh_min
        test = [test;1];
    else
        test = [test;2];
    end
end

count_LDA = 0;
for i=1:test_size*3
    if test(i) == answer(i)
        count_LDA = count_LDA + 1;
    end
end
disp("Test 1: 10 modes")
disp(count_LDA) % 7 right
disp(count_LDA / (test_size*3)) % 38.89%

%% 20 modes: ari < bryce < luke
% bryce < luke
t1 = tr_size;
t2 = 1;
while sort_luke(t1) > sort_bryce(t2)
    t1 = t1-1;
    t2 = t2+1;
end
thresh_min = (sort_luke(t1)+sort_bryce(t2))/2';

% ari < luke
t1 = tr_size;
t2 = 1;
while sort_ari(t1) > sort_luke(t2)
    t1 = t1-1;
    t2 = t2+1;
end
thresh_max = (sort_ari(t1)+sort_luke(t2))/2';
test = [];
for i=1:test_size*3
    if pval(i) > thresh_max
        test = [test;3];
    elseif pval(i) < thresh_min
        test = [test;1];
    else
        test = [test;2];
    end
end

count_LDA = 0;
wrong = [];
for i=1:test_size*3
    if test(i) == answer(i)
        count_LDA = count_LDA + 1;
    else
       wrong = [wrong; i];
    end
end
disp("Test 1: 20 modes")
disp(count_LDA) % 9 right
disp(count_LDA / (test_size*3)) % 50%

%% build matrix
counts = zeros(3,3);
for i=1:test_size*3
    t = test(i);
    a = answer(i);
    for j=1:3
        for k=1:3
            if (t==j) && (a==k)
                counts(j,k) = counts(j,k) + 1;
            end
        end
    end
end

%% 30 modes: ari < bryce < luke
% bryce < luke
t1 = tr_size;
t2 = 1;
while sort_bryce(t1) > sort_luke(t2)
    t1 = t1-1;
    t2 = t2+1;
end
thresh_min = (sort_bryce(t1)+sort_luke(t2))/2';

% ari < bryce
t1 = tr_size;
t2 = 1;
while sort_ari(t1) > sort_bryce(t2)
    t1 = t1-1;
    t2 = t2+1;
end
thresh_max = (sort_ari(t1)+sort_bryce(t2))/2';
test = [];
for i=1:test_size*3
    if pval(i) > thresh_max
        test = [test;2];
    elseif pval(i) < thresh_min
        test = [test;1];
    else
        test = [test;3];
    end
end

count_LDA = 0;
for i=1:test_size*3
    if test(i) == answer(i)
        count_LDA = count_LDA + 1;
    end
end
disp("Test 1: 30 modes")
disp(count_LDA) % 9 right
disp(count_LDA / (test_size*3)) % 50%



