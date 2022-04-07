import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier   #MLPClassifier（多层感知器分类器）
from sklearn.model_selection import train_test_split
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier

#adv_data = pd.read_excel(r'E:\LJ\数据1.xlsx', sheet_name='Sheet2')
adv_data = pd.read_excel(r'E:\LJ\数据1.xlsx', sheet_name='Sheet2')
#new_adv_data = adv_data.iloc[:, 1:]
X = adv_data.iloc[:, 2:10]
y = adv_data['lei']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=100)
#clf = MLPClassifier(alpha=1e-5,  hidden_layer_sizes=(50, 50), max_iter=800, random_state=1)

#clf = svm.SVC(kernel='linear', gamma=1)
clf = RandomForestClassifier(n_estimators=57)
#clf = KNeighborsClassifier(weights="distance", n_neighbors=4, p=1)
#clf = GradientBoostingClassifier(n_estimators=10)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
score = precision_score(y_test, y_pred, average='micro')

print(y_test)
print(y_pred)
print('score=', score)


sns.set()
C2 = confusion_matrix(y_test, y_pred)
C2 = C2.astype('float') / C2.sum(axis=1)[:, np.newaxis]
sns.heatmap(C2, annot=True, cmap='Blues')
plt.xlabel('Predicted label') #x 轴
plt.ylabel('True label') #y 轴
#plt.xticks([0.5, 1.5, 2.5, 3.5], (' SS', ' JS', ' TI', ' LP'))
#plt.yticks([0.5, 1.5, 2.5, 3.5], (' SS', ' JS', ' TI', ' LP'))
plt.xticks(np.arange(0.5, 6.5, 1),('MoS','SS','IF','DS','MaS','Standard'), fontsize=8)
plt.yticks(np.arange(0.5, 6.5, 1),('MoS','SS','IF','DS','MaS','Standard'), fontsize=8)
plt.title('Confusion matrix') #标题
plt.show()
