close all;
clear all; 
clc;

%%

data_dir = 'C:\Users\andre\Desktop\Salerno\projects\gpt\data';
pred_dir = 'C:\Users\andre\Desktop\Salerno\projects\gpt\data\predictors';

time_windows = {'15s', '30s','45s', '60s','75s','90s'};
for i = 1:length(time_windows)
    
    time_window = time_windows{i};
    disp(['Current time window: ' time_window]) 

    saliency_preds = load([pred_dir,'\',time_window,'\sliding_window_design_matrices.mat']);

    saliency = saliency_preds.surp;
    duration = saliency_preds.dur;
    freq = saliency_preds.freq;
    env = saliency_preds.env;
    vifs = [];
    for i=1:size(saliency,2)

        mat = [saliency(:,i),duration(:,i),freq(:,i), env(:,i)];
        vifs = [vifs;vif(mat)];

    end


    disp(mean(vifs))
end

%% Surprisal
data_dir = 'C:\Users\andre\Desktop\Salerno\projects\gpt\data';
pred_dir = 'C:\Users\andre\Desktop\Salerno\projects\gpt\data\predictors';

time_windows = {'15s', '30s','45s', '60s','75s','90s'};
for i = 1:length(time_windows)
    
    time_window = time_windows{i};
    disp(['Current time window: ' time_window]) 

    surp_preds = readmatrix([pred_dir,'\',time_window,'\surprisal_full_timecourse_preds.csv']);

    disp(vif([surp_preds(:,1), surp_preds(:,3), surp_preds(:,4)]))
   
end

