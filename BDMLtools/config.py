#设定分箱全局参数
#digit:数值分箱分箱点小数点后精度，默认3
#combine_ratio:kmeans分箱合并阈值,在bin_limit=1的情况下,阈值越大合并的箱数越多
#max_bin:分箱算法的预分箱数
#max_iters:决策树分箱树的最大递归次数
#chi2_tol:卡方分箱合并分箱的卡方显著性阈值
#levels:分类变量若水平大于levels将被剔除不进行分箱
#regularization:默认True，输出ks、lift、auc指标。此类指标更适合评分类变量。注意此指标会受到分箱切分点、正则参数的影响
#[bin_params]
digit=3
combine_ratio=0.1 
max_bin=50 
max_iters=100
chi2_tol=0.1 

#shap_color_bar:默认False，shapsummaryplot中的颜色条是否显示，默认不显示以防止错误
#[plot_params]
shap_color_bar=False
     
    
 