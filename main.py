import seaborn as sns
from dataset import DataProcessor
from model import DeepCovidClassifier
from sklearn import metrics
from matplotlib import pyplot as plt

if __name__ == '__main__':
  # Grab our preprocessed dataset
  processor = DataProcessor()
  processor.gen_dataset()

  # Init CNN with weights
  cnn = DeepCovidClassifier(processor).create(summary=True, weights='./models/covidCNN.h5', training=False)

  # Predict on holdout data
  x_test, y_test = processor.get_test_data()
  y_pred = cnn.predict(x_test)
  y_pred = (y_pred > 0.5).astype('int')

  print(f"\tAccuracy: {round(metrics.accuracy_score(y_test, y_pred) * 100, 2)}%")
  print(metrics.classification_report(y_test, y_pred))

  p = sns.heatmap(metrics.confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='YlGnBu')
  p.set_xticklabels(['COVID -ve', 'COVID +ve'])
  p.set_yticklabels(['COVID -ve', 'COVID +ve'])
  plt.xlabel('Predicted')
  plt.ylabel('Truth')
  plt.show()