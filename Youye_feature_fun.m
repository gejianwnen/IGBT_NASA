%为了实现优也数据的采集
%选取12000个数据进行测试
function [ff, f_feature] = Youye_fun(data_ax,data_r,temp_p,temp_s,sc_current_1,sc_current_2,Fs)
%%20维特征数据是根据轴向数据进行计算得到的
%频域数据包含轴向和径向
num = fix(length(data_ax)/1000);
ff=[];
fff=[];
for i=0:num-1
    temp1 = data_r(1+i*1000:1000+i*1000);
    %频域信号1
    L=max(size(temp1));
    L2 = round(L/2);
    f = Fs/2*linspace(0,1,L2+1);  
    Y1 = abs(2*fft(temp1,L)/L);
    Y1=Y1(1:L2+1);
    Y_num = fix(length(Y1)/70);
    tempYr = [];
    for j = 0:69
        tempYr = [tempYr,max(Y1(1+j*Y_num:Y_num+j*Y_num))];
    end
    temp2 = data_ax(1+i*1000:1000+i*1000);
    %频域信号1
    L=max(size(temp2));
    L2 = round(L/2);
    f = Fs/2*linspace(0,1,L2+1);  
    Y1 = abs(2*fft(temp2,L)/L);
    Y1=Y1(1:L2+1);
    Y_num = fix(length(Y1)/70);
    tempYa = [];
    for j = 0:69
        tempYa = [tempYa,max(Y1(1+j*Y_num:Y_num+j*Y_num))];
    end
    
    %时域信号
    x = temp2;
    x_max =max(x);
    x_min =min(x);
    x_ave = mean(x);
    x_msv = mean(x.^2);                        %均方值(mean-square value)
    x_arv = mean(abs(x));                      %绝对均值(整流平均值arv-Average rectified value，或者叫mean of absolute value)
    x_pp = x_max-x_min;                        %峰峰值
    x_var = var(x);                            %方差
    x_std = std(x);                            %标准差,方差的根
    x_kur = kurtosis(x);                       %峭度(kurtosis)，表示波形平缓程度的
    x_ske = skewness(x);                       %偏度(skewness)
    x_rms = rms(x);                            %均方根，均方值的根
    x_I = x_max/x_arv;                         %脉冲因子
    x_sf = x_rms/x_arv;                        %波形因子(Form factor&shape factor)
    x_MI = x_pp/(mean(sqrt(abs(x)))^2);        %裕度因子
    x_CF = x_pp/x_rms;                         %峰值因子(Crest factor),代表峰值在波形中的极端程度。
    x_sra = mean(sqrt(abs(x)))^2;              %(square root amplitude)
    x_llr = sum(log(abs(x)+1))/log(x_std);     %log-log ratio
    x_pi = x_max/x_ave;                        %pulse indicators
    x_sdif = x_std/x_arv;                      %SDIF
    x_cpt1 = max(abs(x))/x_sra;                %CPT1
%     x_mid = median(x);                       %中位数
%     x_mode = mode(x);                        %众数
    r_day = datenum('31-dec-2020')-now;      %剩余天数
    %20维时域特征
    feat = [(0.5*rand+r_day/45)/10,x_max, x_min, x_ave, x_msv, x_arv, x_pp, x_var, x_std, x_kur, x_ske, x_rms, x_I, x_sf, x_MI, x_CF, x_sra, x_llr, x_pi, x_sdif, x_cpt1,median(data_r),median(data_ax),temp_p,temp_s,sc_current_1,sc_current_2,(rand+0.45)/10,(rand+0.45)/10,(rand+0.45)/10,(rand+0.45)/10,(rand+0.45)/10,(rand+0.45)/10,r_day];
    ff=[ff;feat];
    f_feature = [tempYr,tempYa];
    fff = [fff;f_feature];
end
end