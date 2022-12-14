clear;
clc;
close all;
addpath('results\');
n_dataset=17;
n_results=20;
n_method=9;
font_size=12;
results_mean=zeros(n_dataset+1,n_results);

%             results(result_seq_i,:)=[error_count_sedumi t_sedumi...
%                                      error_count_mosek  t_mosek...
%                                      error_count_cdcs8  t_cdcs8...
%                                      error_count_bcr    t_bcr...
%                                      error_count_sdcut  t_sdcut...
%                                      error_count_cdcs20 t_cdcs20...
%                                      err_count_gdpa     t_gdpa...
%                                      err_count_glrbox   t_glrbox...
%                                      err_count_glr      t_glr...
%                                      error_count_sns t_sns];

for dataset_i=1:n_dataset
    [dataset_str] = get_dataset_name(dataset_i);
    result_str=['results_' dataset_str '_min_max_scaling_aaai23_I_eg.mat'];
    load(result_str);
    results_mean(dataset_i,:)=mean(results);   
end
results_mean(dataset_i+1,:)=mean(results_mean(1:dataset_i,:));


results_mean=results_mean(:,[1:16 19:20]); % remove GLR


results_err=results_mean(:,1:2:end-1);
results_time=results_mean(:,2:2:end);

datasize_order=zeros(n_dataset,1);
for dataset_i=1:17
    [dataset_str,read_data] = get_data_quiet(dataset_i);
    label=read_data(:,end);
    if dataset_i~=17
    K=5; % 5-fold
    else
    K=1;   
    end
    rng(0);
    indices = crossvalind('Kfold',label,K); % K-fold cross-validation
    read_data_i=read_data(indices==1,:);
    datasize_order(dataset_i)=size(read_data_i,1);
end

[datasize_order_value,datasize_order_idx]=sort(datasize_order);
datasize_order_idx=[datasize_order_idx; n_dataset+1];

results_err=results_err(datasize_order_idx,:);
results_time=results_time(datasize_order_idx,:);

method_name=["SeDuMi (8)" ...
    'MOSEK (8)' ...
    'CDCS (8)' ...
    'BCR' ...
    'SDcut' ...
    'CDCS (20)' ...
    '\color{black}\bfGDPA' ...
    'GLR-box' ...
    'SNS'];

names = {'australian'; 'breast-cancer'; 'diabetes';...
    'fourclass'; 'german'; 'haberman';...
    'heart'; 'ILPD'; 'liver-disorders';...
    'monk1'; 'pima'; 'planning';...
    'voting'; 'WDBC'; 'sonar';...
    'madelon'; 'colon-cancer'; '\color{black}\bfavg.'};

names=names(datasize_order_idx);

ncolors = distinguishable_colors(n_method);
method_ii=[1 2 3 6 4 5 8 9 7];
figure();hold on;
for i=1:n_method
    if i<=4
        plot(results_err(:,method_ii(i)),...
            'LineStyle','none',...
            'LineWidth',1,...
            'Marker','+',...
            'color',ncolors(method_ii(i),:),'DisplayName',num2str(method_name(method_ii(i))));
    elseif i==5
        plot(results_err(:,method_ii(i)),...
            'LineStyle','none',...
            'LineWidth',1,...
            'Marker','s',...
            'color',ncolors(method_ii(i),:),'DisplayName',num2str(method_name(method_ii(i))));
    elseif i==6
        plot(results_err(:,method_ii(i)),...
            'LineStyle','none',...
            'LineWidth',1,...
            'Marker','p',...
            'color',ncolors(method_ii(i),:),'DisplayName',num2str(method_name(method_ii(i))));
    elseif i==9
        plot(results_err(:,method_ii(i)),...
            'LineStyle','none',...
            'LineWidth',1,...
            'Marker','x',...
            'color',ncolors(method_ii(i),:),'DisplayName',num2str(method_name(method_ii(i))));
    else
        plot(results_err(:,method_ii(i)),...
            'LineStyle','none',...
            'LineWidth',1,...
            'Marker','o',...
            'color',ncolors(method_ii(i),:),'DisplayName',num2str(method_name(method_ii(i))));
    end
end

ylabel('error rate (%)', 'FontSize', font_size);
set(gca,'fontname','times', 'FontSize', font_size) 
xlim([1 n_dataset+1]);
set(gca,'xtick',(1:n_dataset+1),'xticklabel',names);xtickangle(90);
ylim([min(vec(results_err)) max(vec(results_err))]);
grid on;
legend;
title('Fig.3 left');

figure();hold on;
for i=1:n_method 
    if i<=4
        plot(results_time(:,method_ii(i)),...
            'LineStyle','none',...
            'LineWidth',1,...
            'Marker','+',...
            'color',ncolors(method_ii(i),:),'DisplayName',num2str(method_name(method_ii(i))));
    elseif i==5
        plot(results_time(:,method_ii(i)),...
            'LineStyle','none',...
            'LineWidth',1,...
            'Marker','s',...
            'color',ncolors(method_ii(i),:),'DisplayName',num2str(method_name(method_ii(i))));
    elseif i==6
        plot(results_time(:,method_ii(i)),...
            'LineStyle','none',...
            'LineWidth',1,...
            'Marker','p',...
            'color',ncolors(method_ii(i),:),'DisplayName',num2str(method_name(method_ii(i))));
    elseif i==9
        plot(results_err(:,method_ii(i)),...
            'LineStyle','none',...
            'LineWidth',1,...
            'Marker','x',...
            'color',ncolors(method_ii(i),:),'DisplayName',num2str(method_name(method_ii(i))));
    else
        plot(results_time(:,method_ii(i)),...
            'LineStyle','none',...
            'LineWidth',1,...
            'Marker','o',...
            'color',ncolors(method_ii(i),:),'DisplayName',num2str(method_name(method_ii(i))));
    end
end

ylabel('runtime (ms)', 'FontSize', font_size);
set(gca,'fontname','times', 'FontSize', font_size)  % Set it to times
xlim([1 n_dataset+1]);
set(gca,'xtick',(1:n_dataset+1),'xticklabel',names);xtickangle(90);
ylim([min(vec(results_time)) max(vec(results_time))]);
grid on;
set(gca, 'YScale', 'log')
legend;
title('Fig.4 left');