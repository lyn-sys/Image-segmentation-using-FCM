close all;
I = imread('../image-sonar/sonar5.png');
rng('default');
[m, n, p] = size(I);
X = reshape(double(I), m*n, p);
k = 3; b = 2;
[C, dist, J] = fcm(X, k, b);
[~, label] = min(dist, [], 2);

figure
imshow(uint8(reshape(C(label, :), m, n, p)))
figure
plot(1:length(J), J, 'r-*'), xlabel('#iterations'), ylabel('objective function')