from scipy.stats import kendalltau

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# sns.set(font_scale=1.1)


plt.rcParams.update({'font.size': 12})

tips = pd.read_csv('review_en.csv')
print(tips.head())
result = tips.corr(method='kendall')


result_1, p  = kendalltau(tips['Suggesting'],tips['Evaluating'])
print(result_1, p)

# ,Questioning,Kindness'])
print(result_1)
print(tips.corr())
# sns.pairplot(tips)

mask = np.zeros_like(result)
mask[np.triu_indices_from(mask)] = True
sns.heatmap(result, annot=True, mask=mask, cmap='vlag')

fig = plt.gcf()
fig.set_size_inches(10, 3.5)

plt.savefig("correlationAnalysis.png", dpi=600)
plt.show()
