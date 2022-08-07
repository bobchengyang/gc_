clear;
clc;
close all;
rng('default');

% cvx_precision low: [ϵ3/8,ϵ1/4,ϵ1/4]
% cvx_precision medium: [ϵ1/2,ϵ3/8,ϵ1/4]
% cvx_precision default: [ϵ1/2,ϵ1/2,ϵ1/4]
% cvx_precision high: [ϵ3/4,ϵ3/4,ϵ3/8]
% cvx_precision best: [0,ϵ1/2,ϵ1/4]

cvx_begin quiet
cvx_precision default
cvx_precision

n_sample=100;
number_of_neighbor=2;
noise_level=1.0;

label=[ones(n_sample/2,1);-ones(n_sample/2,1)];

rng(0);
signal=label+randn(n_sample,1)*noise_level;
figure();plot(1:n_sample,label);
title('groundtruth');
grid on;
ylim([-2 2]);
figure();plot(1:n_sample,signal);
title('corrupted');
grid on;
ylim([-5 5]);
[L,La]=Q(n_sample,number_of_neighbor,signal);

%% SDP primal
disp('SDP primal ====================================================================');
clear dB dA;
% dL = [-L zeros(n_sample,1); zeros(1,n_sample) 0];
dL = [1 signal'; signal La];
b_ind=1;
[x_pred_sedumi,error_count_sedumi,obj_sedumi_xlx,obj_sedumi,t_sedumi] = ...
    sedumi_sdp_prime_uc(dL,n_sample,b_ind,label);
disp(['SDP primal error_count: ' num2str(error_count_sedumi/n_sample*100) '%']);
disp(['SDP primal xlx obj: ' num2str(obj_sedumi_xlx)]);
disp(['SDP primal obj: ' num2str(obj_sedumi)]);

%% SDP dual
ei_l = eye(n_sample+1);   %% two identity matrices
for i=1:n_sample+1
    dA(:,:,i) = diag(ei_l(:,i)); % eq. (7), where dA is a stacked matrix where each of them has only 1 non-zero entry
end
disp('SDP dual ====================================================================');
[x_pred_sedumi_dual,err_count_sedumi_dual,obj_sedumi_dual_xlx,obj_sedumi_dual,t_sedumi_dual] = ...
    sedumi_sdp_dual_uc(dL,dA,n_sample,b_ind,label);
disp(['SDP dual error_count: ' num2str(err_count_sedumi_dual/n_sample*100) '%']);
disp(['SDP dual xlx obj: ' num2str(obj_sedumi_dual_xlx)]);
disp(['SDP dual obj: ' num2str(obj_sedumi_dual)]);

%% modified SDP dual
disp('SDP dual modified ====================================================================');
epsilon=0;
[x_pred_sedumi_dual_modified,err_count_sedumi_dual_modified,obj_sedumi_dual_modified_xlx,obj_sedumi_dual_modified,t_sedumi_dual_modified,dy,dz] = ...
    sedumi_sdp_dual_modified_uc_(dL,n_sample,b_ind,label,epsilon);
disp(['SDP dual modified error_count: ' num2str(err_count_sedumi_dual_modified/n_sample*100) '%']);
disp(['SDP dual modified xlx obj: ' num2str(obj_sedumi_dual_modified_xlx)]);
disp(['SDP dual modified obj: ' num2str(obj_sedumi_dual_modified)]);

%% GDPA (modified SDP dual where the PSD constraint is replaced with linear constraints)
disp('GDPA ====================================================================');
[obj_gdpa_xlx,obj_gdpa,err_count_gdpa] = gdpa_tsp_main_uc_(label,b_ind,n_sample,dL,dy,dz,epsilon);
disp(['GDPA error_count: ' num2str(err_count_gdpa/n_sample*100) '%']);
disp(['GDPA xlx obj: ' num2str(obj_gdpa_xlx)]);
disp(['GDPA obj: ' num2str(obj_gdpa)]);

%% GLR (spectral method)
disp('GLR ====================================================================');
[err_count_glr,obj_glr,t_glr] = ...
    glr_closed_form_uc(dL,n_sample,b_ind,label);
disp(['GLR error_count: ' num2str(err_count_glr/n_sample*100) '%']);
disp(['GLR obj: ' num2str(obj_glr)]);   

%% SNS (non-SDP)
addpath('BQP-master\BQP-master\');
disp('SNS ====================================================================');
[error_count_sns,obj_xlx,obj_sns,t_sns] = sns_uc(dL,n_sample,b_ind,label);
disp(['SNS error_count: ' num2str(error_count_sns/n_sample*100) '%']);
disp(['SNS xlx obj: ' num2str(obj_xlx)]); 
disp(['SNS obj: ' num2str(obj_sns)]);  
