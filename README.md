**README.md**

# Skin Condition Classifier

This project provides a comprehensive framework for classifying skin conditions using a convolutional neural network (CNN), transfer learning with VGG16, and machine learning models such as SVM and KNN. The system supports data augmentation, training history visualization, and performance evaluation through metrics and confusion matrices.

## Features
- **CNN Model**: Custom-built CNN architecture with dropout, batch normalization, and L2 regularization.
- **Transfer Learning**: Leverages the VGG16 model as a feature extractor.
- **Machine Learning**: Integrates SVM and KNN models for classification using extracted features.
- **Visualization**: Plots training history and confusion matrices for better interpretability.
- **Callbacks**: Includes early stopping and learning rate reduction.

## Prerequisites
- Python 3.7+
- TensorFlow/Keras
- Scikit-learn
- Matplotlib
- Numpy

Install dependencies using:
```bash
pip install tensorflow keras scikit-learn matplotlib numpy
```

## Usage
1. Initialize the classifier with training and testing directories:
   ```python
   classifier = SkinConditionClassifier(train_dir='/content/train', test_dir='/content/test')
   ```

2. Train the CNN model:
   ```python
   classifier.build_cnn_model()
   classifier.train_cnn()
   ```

3. Evaluate the CNN model:
   ```python
   classifier.evaluate_model(classifier.cnn_model)
   classifier.plot_confusion_matrix(classifier.cnn_model)
   ```

4. Train the Transfer Learning model:
   ```python
   classifier.build_transfer_learning_model()
   classifier.train_transfer_learning()
   ```

5. Evaluate the Transfer Learning model:
   ```python
   classifier.evaluate_model(classifier.transfer_model)
   classifier.plot_confusion_matrix(classifier.transfer_model)
   ```

6. Extract features and train ML models (SVM and KNN):
   ```python
   train_features, train_labels, test_features, test_labels = classifier.extract_features_for_ml(classifier.cnn_model)
   classifier.train_svm(train_features, train_labels, test_features, test_labels)
   classifier.train_knn(train_features, train_labels, test_features, test_labels)
   ```


