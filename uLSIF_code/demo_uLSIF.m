% demo_uLSIF.m
%
% (c) Masashi Sugiyama, Department of Compter Science, Tokyo Institute of Technology, Japan.
%     sugi@cs.titech.ac.jp,     http://sugiyama-www.cs.titech.ac.jp/~sugi/software/uLSIF/
clc
clear all

rand('state',0);
randn('state',0);

%%%%%%%%%%%%%%%%%%%%%%%%% Generating data
d=1;

dataset=1;
switch dataset
 case 1
  n_de=100;
  n_nu=100;
  mu_de=1;
  mu_nu=1;
  %p_de = 0.5;
  %p_nu = 0.1;
  sigma_de=1/2;
  sigma_nu=1/8;
  legend_position='northeast';
 case 2
  n_de=200;
  n_nu=1000;
  mu_de=1;
  mu_nu=2;
%   p_de = 0.6;
%   p_nu = 0.3;
  sigma_de=1/2;
  sigma_nu=1/4;
  legend_position='northwest';
end
% x_de = binornd(mu_de, p_de, 1, n_de);
% x_nu = binornd(mu_nu, p_nu, 1, n_nu);
x_de=mu_de+sigma_de*randn(d,n_de);
x_nu=mu_nu+sigma_nu*randn(d,n_nu);

% x_disp = linspace(0, max(mu_de, mu_nu) + 2, 100);
x_disp=linspace(-0.5,3,100);
% p_de_x_disp = binopdf(x_disp, mu_de, p_de);
% p_nu_x_disp = binopdf(x_disp, mu_nu, p_nu);
p_de_x_disp=pdf_Gaussian(x_disp,mu_de,sigma_de);
p_nu_x_disp=pdf_Gaussian(x_disp,mu_nu,sigma_nu);
w_x_disp=p_nu_x_disp./p_de_x_disp;

% p_de_x_de = binopdf(x_de, mu_de, p_de);
% p_nu_x_de = binopdf(x_de, mu_nu, p_nu);
p_de_x_de=pdf_Gaussian(x_de,mu_de,sigma_de);
p_nu_x_de=pdf_Gaussian(x_de,mu_nu,sigma_nu);
w_x_de=p_nu_x_de./p_de_x_de;

%%%%%%%%%%%%%%%%%%%%%%%%% Estimating density ratio
%[wh_x_de,wh_x_disp]=uLSIF(x_de,x_nu,x_disp);
[wh_x_de,wh_x_disp]=uLSIF(x_de,x_nu,x_disp,[],[],[],0);



figure(1)
clf
hold on
set(gca,'FontName','Helvetica')
set(gca,'FontSize',12)
plot(x_disp,p_de_x_disp,'b-','LineWidth',2)
plot(x_disp,p_nu_x_disp,'k-','LineWidth',2)
legend({'p_{de}(x)','p_{nu}(x)'},'Location',legend_position)
xlabel('x')
set(gcf,'PaperUnits','centimeters');
set(gcf,'PaperPosition',[0 0 12 9]);
print('-depsc',sprintf('density%g',dataset))

figure(2)
clf
hold on
set(gca,'FontName','Helvetica')
set(gca,'FontSize',12)
plot(x_disp,w_x_disp,'r-','LineWidth',2)
plot(x_disp,wh_x_disp,'g-','LineWidth',2)
plot(x_de,wh_x_de,'bo','LineWidth',1,'MarkerSize',8)
legend({'w(x)','w-hat(x)','w-hat(x^{de})'},'Location',legend_position)
xlabel('x')
set(gcf,'PaperUnits','centimeters');
set(gcf,'PaperPosition',[0 0 12 9]);
print('-depsc',sprintf('importance%g',dataset))





