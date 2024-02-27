import os
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report
from skimage.io import imread
from skimage.transform import resize
import matplotlib.pyplot as plt
import pandas as pd


def load_dataset(root_folder):
    images = []
    labels = []
    class_mapping = {}

    for class_label, class_name in enumerate(os.listdir(root_folder)):
        class_mapping[class_label] = class_name
        class_folder = os.path.join(root_folder, class_name)

        for filename in os.listdir(class_folder):
            image_path = os.path.join(class_folder, filename)
            image = imread(image_path)
            resize_image = resize(image, (64, 64, 3), anti_aliasing=True)
            images.append(resize_image.flatten())
            labels.append(class_label)

    return np.array(images), np.array(labels), class_mapping

# Function to save results to a CSV file
def save_results(algorithm, y_true, y_pred, class_mapping):
    results_df = pd.DataFrame({'True Label': [class_mapping[label] for label in y_true],
                    'Predicted Label': [class_mapping[label] for label in y_pred]})

    results_folder = './results'
    if not os.path.exists(results_folder):
        os.makedirs(results_folder)

    results_file = os.path.join(results_folder, f'{algorithm}_results.csv')
    results_df.to_csv(results_file, index=False)
    print(f'Results saved for {algorithm} at {results_file}')

# Function to save test images with predictions
def save_test_images(algorithm, X_test, y_true, y_pred, class_mapping, probabilities):
    results_folder = './results'
    if not os.path.exists(results_folder):
        os.makedirs(results_folder)

    for idx, (true_label, pred_label, prob_dist) in enumerate(zip(y_true, y_pred, probabilities)):
        image = X_test[idx].reshape(64, 64, -1) # Reshape flattened image
        plt.imshow(image)
        plt.title(f'True Label: {class_mapping[true_label]}\nPredicted Label: {class_mapping[pred_label]}')
        plt.savefig(os.path.join(results_folder, f'{algorithm}_test_image_{idx}.png'))
        plt.close()

        # Plot probability distribution
        plt.bar(class_mapping.values(), prob_dist, color='blue')
        plt.title(f'Probability Distribution - Test Image {idx}')
        plt.xlabel('Class')
        plt.ylabel('Probability')
        plt.savefig(os.path.join(results_folder, f'{algorithm}_probability_distribution_{idx}.png'))
        plt.close()

    print(f'Test images saved for {algorithm} in {results_folder}')

# Load your dataset
dataset_folder = "./dataset" # Change this to the path of your dataset folder
X_train, y_train, class_mapping = load_dataset(dataset_folder)

test_folder = "./test"
X_test, y_test, class_mapping2 = load_dataset(test_folder)

valid_folder = "./valid"
X_valid, y_valid, class_mapping3 = load_dataset(valid_folder)

# Example 1: k-Nearest Neighbors (k-NN)
# Initialize k-NN classifier
knn_classifier = KNeighborsClassifier(n_neighbors=5)
# Train the model
knn_classifier.fit(X_train, y_train)
# Make predictions on the test set
y_test_pred_knn = knn_classifier.predict(X_test)
# Make predictions on the validation set
y_valid_pred_knn = knn_classifier.predict(X_valid)
# Save results and test images for k-NN
save_results('knn', y_test, y_test_pred_knn, class_mapping)
probabilities_knn = knn_classifier.predict_proba(X_test)
save_test_images('knn', X_test, y_test, y_test_pred_knn, class_mapping, probabilities_knn)

# Example 2: Naive Bayes (Gaussian Naive Bayes)
# Initialize Naive Bayes classifier
nb_classifier = GaussianNB()
# Train the model
nb_classifier.fit(X_train, y_train)
# Make predictions on the test set
y_test_pred_nb = nb_classifier.predict(X_test)
# Make predictions on the validation set
y_valid_pred_nb = nb_classifier.predict(X_valid)
# Save results and test images for Naive Bayes
save_results('naive_bayes', y_test, y_test_pred_nb, class_mapping)
probabilities_nb = nb_classifier.predict_proba(X_test)
save_test_images('naive_bayes', X_test, y_test, y_test_pred_nb, class_mapping, probabilities_nb)

# The accuracy score for K-NN method
score = accuracy_score(y_test_pred_knn, y_test)
print('{}% of test samples were correctly classified -- k-NN'.format(str(score * 100)))

# The accuracy score for K-NN method for validation set
score = accuracy_score(y_valid_pred_knn, y_valid)
print('{}% of validation samples were correctly classified -- k-NN'.format(str(score * 100)))

# The accuracy score for Naive Bayes method
score = accuracy_score(y_test_pred_nb, y_test)
print('{}% of test samples were correctly -- NB'.format(str(score * 100)))

# The accuracy score for Naive Bayes method for validation set
score = accuracy_score(y_valid_pred_nb, y_valid)
print('{}% of validation samples were correctly classified -- NB'.format(str(score * 100)))

# Cross-validation for both methods
folds = 5
model_accuracy = cross_val_score(knn_classifier, X_train, y_train, cv=folds, scoring='accuracy')
print(f'Average accuracy for k-NN: {model_accuracy.mean() * 100:.2f}%')

folds = 5
model_accuracy = cross_val_score(nb_classifier, X_train, y_train, cv=folds, scoring='accuracy')
print(f'Average accuracy for NB: {model_accuracy.mean() * 100:.2f}%')