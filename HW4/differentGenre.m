%%
clear all; close all; clc; 
%% TEST 3
alldata = [];
artists = ["morgan_wallen", "luke_bryan", "eric_church", "ariana_grande","camila_cabello", "ed_sheeran", "mackelmore", "bryce_vine", "travis_scott"];
myDir = '/Users/audrey/Desktop/Music/';
songs = 20;
for i = 1:length(artists)
    for j = 1:songs % 20 5" clips from each artist
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
songs = 60; % per genre
singers = u*v';
q = randperm(songs);
qq = q;
tr_size = 45;
test_size = songs - tr_size;
%%
features = 10;
country = singers(1:features,1:songs);
pop = singers(1:features,songs+1:songs*2);
rap = singers(1:features,(songs*2)+1:end);

% training sets
country_train = country(:,q(1:tr_size));
pop_train = pop(:,q(1:tr_size));
rap_train = rap(:,q(1:tr_size));

% test sets
country_test = country(:,q(tr_size+1:end));
pop_test = pop(:,q(tr_size+1:end));
rap_test = rap(:,q(tr_size+1:end));

m1 = mean(country_train,2);
m2 = mean(pop_train,2);
m3 = mean(rap_train,2);
m_all = mean(singers);

% Within class
Sw = 0;
for i=1:tr_size
    Sw = Sw + (country_train(:,i)-m1)*(country_train(:,i)-m1)';
end
for i=1:tr_size
    Sw = Sw + (pop_train(:,i)-m2)*(pop_train(:,i)-m2)';
end
for i=1:tr_size
    Sw = Sw + (rap_train(:,i)-m3)*(rap_train(:,i)-m3)';
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

vcountry = w'*country_train;
vpop = w'*pop_train;
vrap = w'*rap_train;

sort_country = sort(vcountry);
sort_pop = sort(vpop);
sort_rap = sort(vrap);

vcountry_test = w'*country_test;
vpop_test = w'*pop_test;
vrap_test = w'*rap_test;
pval = [vcountry_test vpop_test vrap_test];
answer = [ones(test_size,1); 2*ones(test_size,1); 3*ones(test_size,1)];

disp(mean([vcountry;vpop;vrap],2));
disp(sort(mean([vcountry;vpop;vrap],2)));

%% 10 modes: country < rap < pop
% Best classifier
% Determine the threshold values
% rap < pop
t1 = tr_size;
t2 = 1;
while sort_rap(t1) > sort_pop(t2)
    t1 = t1-1;
    t2 = t2+1;
end
thresh_max = (sort_rap(t1)+sort_pop(t2))/2';

% country < rap
t1 = tr_size;
t2 = 1;
while sort_country(t1) > sort_rap(t2)
    t1 = t1-1;
    t2 = t2+1;
end
thresh_min = (sort_country(t1)+sort_rap(t2))/2';

% Test on test set
% look at values and classify
% country 1 < rap 3 < pop 2
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
disp("Test 3: 10 modes")
disp(count_LDA) % 16
disp(count_LDA / (test_size*3)) % 35.56%
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


%% 20 modes: country < rap < pop
% Determine the threshold values
% rap < pop
t1 = tr_size;
t2 = 1;
while sort_rap(t1) > sort_pop(t2)
    t1 = t1-1;
    t2 = t2+1;
end
thresh_max = (sort_rap(t1)+sort_pop(t2))/2';

% country < rap
t1 = tr_size;
t2 = 1;
while sort_country(t1) > sort_rap(t2)
    t1 = t1-1;
    t2 = t2+1;
end
thresh_min = (sort_country(t1)+sort_rap(t2))/2';

% Test on test set
% look at values and classify
% country 1 < rap 3 < pop 2
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
disp("Test 3: 20 modes")
disp(count_LDA) % 10
disp(count_LDA / (test_size*3)) % 22.22%

%% 30 modes: country < rap < pop
% Determine the threshold values
% rap < pop
t1 = tr_size;
t2 = 1;
while sort_rap(t1) > sort_pop(t2)
    t1 = t1-1;
    t2 = t2+1;
end
thresh_max = (sort_rap(t1)+sort_pop(t2))/2';

% country < rap
t1 = tr_size;
t2 = 1;
while sort_country(t1) > sort_rap(t2)
    t1 = t1-1;
    t2 = t2+1;
end
thresh_min = (sort_country(t1)+sort_rap(t2))/2';

% Test on test set
% look at values and classify
% country 1 < rap 3 < pop 2
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
disp("Test 3: 30 modes")
disp(count_LDA) % 13
disp(count_LDA / (test_size*3)) % 28.9%