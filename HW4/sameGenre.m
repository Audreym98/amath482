%%
clear all; close all; clc; 
%% TEST 2
% add more samples (14 training, 6 test)
alldata = [];
artists = ["morgan_wallen", "luke_bryan", "eric_church"];
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
singers = u*v';
q = [16 15 12 14 13 5 2 6 8 17 18 1 4 20 9 11 7 10 19 3];
% transpose alldata for classify
%%
features = 20;
tr_size = 14;
test_size = songs - tr_size;

mo = singers(1:features,1:20);
luke = singers(1:features,21:40);
eric = singers(1:features,41:end);

% training sets
mo_train = mo(:,q(1:tr_size));
luke_train = luke(:,q(1:tr_size));
eric_train = eric(:,q(1:tr_size));

% test sets
mo_test = mo(:,q(tr_size+1:end));
luke_test = luke(:,q(tr_size+1:end));
eric_test = eric(:,q(tr_size+1:end));

m1 = mean(mo_train,2);
m2 = mean(luke_train,2);
m3 = mean(eric_train,2);
m_all = mean(singers);

% Within class
Sw = 0;
for i=1:tr_size
    Sw = Sw + (mo_train(:,i)-m1)*(mo_train(:,i)-m1)';
end
for i=1:tr_size
    Sw = Sw + (luke_train(:,i)-m2)*(luke_train(:,i)-m2)';
end
for i=1:tr_size
    Sw = Sw + (eric_train(:,i)-m3)*(eric_train(:,i)-m3)';
end

% Between class
Sb = 0;
Sb = Sb + (m1-m_all)*(m1-m_all)'*tr_size;
Sb = Sb + (m2-m_all)*(m2-m_all)'*tr_size;
Sb = Sb + (m3-m_all)*(m3-m_all)'*tr_size;
% Project onto single vector
[V2,D] = eig(Sb,Sw);
[~,ind] = max(abs(diag(D)));
w = V2(:,ind); w = w/norm(w,2);

vmo = w'*mo_train;
vluke = w'*luke_train;
veric = w'*eric_train;

vmo_test = w'*mo_test;
vluke_test = w'*luke_test;
veric_test = w'*eric_test;
pval = [vmo_test vluke_test veric_test];
answer = [ones(test_size,1); 2*ones(test_size,1); 3*ones(test_size,1)];

disp(mean([vmo;vluke;veric],2));
disp(sort(mean([vmo;vluke;veric],2)));

sort_mo = sort(vmo);
sort_luke = sort(vluke);
sort_eric = sort(veric);
%% 10 modes
% Determine the threshold values
%  mo < luke < eric
% luke < eric
t1 = tr_size;
t2 = 1;
while sort_luke(t1) > sort_eric(t2)
    t1 = t1-1;
    t2 = t2+1;
end
thresh_max = (sort_luke(t1)+sort_eric(t2))/2';

% mo < luke
t1 = tr_size;
t2 = 1;
while sort_mo(t1) > sort_luke(t2)
    t1 = t1-1;
    t2 = t2+1;
end
thresh_min = (sort_mo(t1)+sort_luke(t2))/2';

%  mo (1) < luke < eric (3)
% look at values and classify
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
disp("Test 2: 10 modes")
disp(count_LDA) % 8
disp(count_LDA / (test_size*3)) % 44%

%% 20 modes
% Determine the threshold values
%  mo < luke < eric
% luke < eric
t1 = tr_size;
t2 = 1;
while sort_luke(t1) > sort_eric(t2)
    t1 = t1-1;
    t2 = t2+1;
end
thresh_max = (sort_luke(t1)+sort_eric(t2))/2';

% mo < luke
t1 = tr_size;
t2 = 1;
while sort_mo(t1) > sort_luke(t2)
    t1 = t1-1;
    t2 = t2+1;
end
thresh_min = (sort_mo(t1)+sort_luke(t2))/2';

%  mo (1) < luke < eric (3)
% look at values and classify
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
disp("Test 2: 20 modes")
disp(count_LDA) % 10
disp(count_LDA / (test_size*3)) % 55.56%
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

%% 30 modes
% Determine the threshold values
%  mo < luke < eric
% luke < eric
t1 = tr_size;
t2 = 1;
while sort_luke(t1) > sort_eric(t2)
    t1 = t1-1;
    t2 = t2+1;
end
thresh_max = (sort_luke(t1)+sort_eric(t2))/2';

% mo < luke
t1 = tr_size;
t2 = 1;
while sort_mo(t1) > sort_luke(t2)
    t1 = t1-1;
    t2 = t2+1;
end
thresh_min = (sort_mo(t1)+sort_luke(t2))/2';

%  mo (1) < luke < eric (3)
% look at values and classify
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
disp("Test 2: 30 modes")
disp(count_LDA) % 8
disp(count_LDA / (test_size*3)) % 44.4%