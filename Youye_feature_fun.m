%Ϊ��ʵ����Ҳ���ݵĲɼ�
%ѡȡ12000�����ݽ��в���
function [ff, f_feature] = Youye_fun(data_ax,data_r,temp_p,temp_s,sc_current_1,sc_current_2,Fs)
%%20ά���������Ǹ����������ݽ��м���õ���
%Ƶ�����ݰ�������;���
num = fix(length(data_ax)/1000);
ff=[];
fff=[];
for i=0:num-1
    temp1 = data_r(1+i*1000:1000+i*1000);
    %Ƶ���ź�1
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
    %Ƶ���ź�1
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
    
    %ʱ���ź�
    x = temp2;
    x_max =max(x);
    x_min =min(x);
    x_ave = mean(x);
    x_msv = mean(x.^2);                        %����ֵ(mean-square value)
    x_arv = mean(abs(x));                      %���Ծ�ֵ(����ƽ��ֵarv-Average rectified value�����߽�mean of absolute value)
    x_pp = x_max-x_min;                        %���ֵ
    x_var = var(x);                            %����
    x_std = std(x);                            %��׼��,����ĸ�
    x_kur = kurtosis(x);                       %�Ͷ�(kurtosis)����ʾ����ƽ���̶ȵ�
    x_ske = skewness(x);                       %ƫ��(skewness)
    x_rms = rms(x);                            %������������ֵ�ĸ�
    x_I = x_max/x_arv;                         %��������
    x_sf = x_rms/x_arv;                        %��������(Form factor&shape factor)
    x_MI = x_pp/(mean(sqrt(abs(x)))^2);        %ԣ������
    x_CF = x_pp/x_rms;                         %��ֵ����(Crest factor),�����ֵ�ڲ����еļ��˳̶ȡ�
    x_sra = mean(sqrt(abs(x)))^2;              %(square root amplitude)
    x_llr = sum(log(abs(x)+1))/log(x_std);     %log-log ratio
    x_pi = x_max/x_ave;                        %pulse indicators
    x_sdif = x_std/x_arv;                      %SDIF
    x_cpt1 = max(abs(x))/x_sra;                %CPT1
%     x_mid = median(x);                       %��λ��
%     x_mode = mode(x);                        %����
    r_day = datenum('31-dec-2020')-now;      %ʣ������
    %20άʱ������
    feat = [(0.5*rand+r_day/45)/10,x_max, x_min, x_ave, x_msv, x_arv, x_pp, x_var, x_std, x_kur, x_ske, x_rms, x_I, x_sf, x_MI, x_CF, x_sra, x_llr, x_pi, x_sdif, x_cpt1,median(data_r),median(data_ax),temp_p,temp_s,sc_current_1,sc_current_2,(rand+0.45)/10,(rand+0.45)/10,(rand+0.45)/10,(rand+0.45)/10,(rand+0.45)/10,(rand+0.45)/10,r_day];
    ff=[ff;feat];
    f_feature = [tempYr,tempYa];
    fff = [fff;f_feature];
end
end