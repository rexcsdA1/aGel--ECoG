clear all;
%% 1. 导入数据Implant_Day

load('SNR_allweek_processed.mat', 'data');

data = data(data.Implant_Day >0, :);



% 将分类变量转换为分类类型
data.ElectrodeType = categorical(data.ElectrodeType);
data.RatID = categorical(data.Rat_ID);
data.ElectrodeID = categorical(data.Electrode_ID);
data.Impedance = double(data.Impedance);
data.Week = double(data.Week);


% 只保留正整数 Week
data = data(data.Week > 0 & mod(data.Week, 1) == 0, :);

%% 3. 设置 aGel(+) 为基线并拟合模型
% 将 aGel(+) 作为基线类别
data.ElectrodeType = reordercats(data.ElectrodeType, {'aGel(+)', 'aGel(-)'});

% 定义混合效应模型
lmeFormula = 'SNR_8~ ElectrodeType*Week+ (1|RatID)+(1|ElectrodeID) ';



% 拟合混合效应模型
lme_hydrogel = fitlme(data, lmeFormula);
coeffs = fixedEffects(lme_hydrogel);
% 显示模型结果
disp('aGel(+) as baseline:');
disp(lme_hydrogel);



%% 4. 生成预测值并进行可视化
% 预测值
[ypred_hydrogel, ~] = predict(lme_hydrogel, data);


% 可视化 aGel(+) 作为基线时的拟合结果
figure; hold on;
idx_Hydrogel = data.ElectrodeType == 'aGel(+)';
idx_PI = data.ElectrodeType == 'aGel(-)';
scatter(data.Week(idx_Hydrogel), data.SNR_8(idx_Hydrogel), 'b', 'DisplayName', 'aGel(+) data', 'MarkerFaceColor', 'b');
scatter(data.Week(idx_PI), data.SNR_8(idx_PI), 'r', 'DisplayName', 'aGel(-) data', 'MarkerFaceColor', 'r');
plot(data.Week(idx_Hydrogel), ypred_hydrogel(idx_Hydrogel), '-b', 'LineWidth', 1.5, 'DisplayName', 'aGel(+) Fit');
plot(data.Week(idx_PI), ypred_hydrogel(idx_PI), '-r', 'LineWidth', 1.5, 'DisplayName', 'aGel(-) Fit');
xlabel('Week');
ylabel('SNR_8');
legend('show');
title('SNR_8 vs. Week');
grid on;
hold off;



% 假设你已经拟合了混合效应模型 lme_hydrogel 或 lme_pi

% 提取效应参数
hydrogel_beta0 = lme_hydrogel.Coefficients.Estimate(1); % Intercept
hydrogel_beta1 = lme_hydrogel.Coefficients.Estimate(2); % EectrodeType_PI
pi_beta2 = lme_hydrogel.Coefficients.Estimate(3); % Week
pi_beta3 = lme_hydrogel.Coefficients.Estimate(4); % ElectrodeType_PI:Week
% 
% 定义 Week 和 ElectrodeType
Week = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16];
ElectrodeType_Hydrogel = 0;
ElectrodeType_PI = 1;

% 计算 Signal_Retention_Hydrogel 和 Signal_Retention_PI
Y_Hydrogel = hydrogel_beta0 + hydrogel_beta1 * ElectrodeType_Hydrogel + pi_beta2 * Week + pi_beta3 * (Week .* ElectrodeType_Hydrogel);
Y_PI = hydrogel_beta0 + hydrogel_beta1 * ElectrodeType_PI + pi_beta2 * Week + pi_beta3 * (Week .* ElectrodeType_PI);

% 计算 Week = 7 时的基准值
Y_Hydrogel_base = Y_Hydrogel(Week == 1);
Y_PI_base = Y_PI(Week == 1);
%%
% 计算相对比值
SNR_Retention_Hydrogel = (Y_Hydrogel / Y_Hydrogel_base*100);
SNR_Retention_PI = (Y_PI / Y_PI_base*100);


% 创建表格
result_table = table(Week', SNR_Retention_Hydrogel', SNR_Retention_PI', ...
                     'VariableNames', {'Week', 'SNR_Retention_aGel(+)', ...
                                       'SNR_Retention_aGel(-)'});
%%
data_hg = data(data.ElectrodeType == 'aGel(+)', :);


yfit_hg = predict(lme_hydrogel, data_hg );

y_hg = data_hg.SNR_8;


r2_hg = 1 - sum((y_hg - yfit_hg ).^2) / sum((y_hg - mean(y_hg)).^2);



