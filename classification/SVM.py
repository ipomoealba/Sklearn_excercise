#-*- coding:utf-8 -*-
import matplotlib.pyplot as plt
from sklearn import svm, datasets, metrics

digits = datasets.load_digits()

# print(digits.keys())
# print(digits.data)


img_and_label = zip(digits.images, digits.target)
# for i, (image, label) in enumerate(img_and_label):
#     # plt.subplot(2, 4, i+1)
#     plt.subplot(2, 4, i)
#     plt.axis('off')
#     plt.imshow(image, cmap=plt.cm.binary)
#     plt.title('Training: '+ str(label))

# plt.show()
s_len = len(digits.images)
data = digits.images.reshape((s_len,-1))

classifier = svm.SVC(gamma=0.001)
# 用前半部份的資料來訓練
classifier.fit(data[:s_len / 2], digits.target[:s_len / 2])

expected = digits.target[s_len / 2:]

#利用後半部份的資料來測試分類器，共 899筆資料
predicted = classifier.predict(data[s_len / 2:])
print("Confusion matrix:\n%s" % metrics.confusion_matrix(expected, predicted))

print("Classification report for classifier %s:\n%s\n"
    % (classifier, metrics.classification_report(expected, predicted)))