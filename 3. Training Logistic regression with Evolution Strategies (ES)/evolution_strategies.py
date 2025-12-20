import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import os
from io import StringIO

np.random.seed(42)


class EvolutionStrategyLogisticRegression:
    def __init__(self, mu=30, lambda_offspring=None, lambda_reg=0.01,
                 max_generations=100):
        self.mu = mu
        self.lambda_offspring = lambda_offspring if lambda_offspring is not None else 7 * mu
        self.lambda_reg = lambda_reg
        self.max_generations = max_generations

        # History tracking
        self.best_fitness_history = []
        self.mean_fitness_history = []
        self.train_accuracy_history = []
        self.test_accuracy_history = []
        self.best_theta = None

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def cross_entropy_loss(self, X, y, theta):
        W = theta[:-1]
        b = theta[-1]
        z = X @ W + b
        y_pred = self.sigmoid(z)

        eps = 1e-15  # avoid log(0)
        y_pred = np.clip(y_pred, eps, 1 - eps)

        n = len(y)
        ce_loss = -1/n * np.sum((y * np.log(y_pred)) +
                                (1-y) * np.log(1-y_pred))

        l2 = self.lambda_reg * np.sum(W ** 2)

        return ce_loss + l2

    def fitness(self, X, y, theta):
        return -self.cross_entropy_loss(X, y, theta)

    def predict(self, X, theta):
        W = theta[:-1]
        b = theta[-1]
        z = X @ W + b
        prob = self.sigmoid(z)
        # Convert True/False array to int values
        predictions = (prob >= 0.5).astype(int)
        return predictions

    def initialize_population(self, d):
        pop = []
        for pop_size in range(self.mu):
            theta = np.random.uniform(-0.1, 0.1, size=d + 1)
            sigma = np.random.uniform(0.01, 0.1, size=d + 1)
            individual = np.concatenate([theta, sigma])
            pop.append(individual)
        return pop

    def mutate(self, individual, n):
        theta = individual[:n]   # First n elements
        sigma = individual[n:]   # Remaining n elements
        tau = 1 / np.sqrt(2*n)
        tau_prime = 1 / np.sqrt(2*np.sqrt(n))
        N_global = np.random.normal(0, 1)
        N_local = np.random.normal(0, 1, size=n)
        sigma_new = sigma * np.exp(tau * N_global + tau_prime * N_local)
        # From: https://numpy.org/doc/stable/reference/generated/numpy.maximum.html
        # Compare two arrays and return a new array containing the element-wise maxima.
        # If one of the elements being compared is a NaN, then that element is returned.
        # If both elements are NaNs then the first is returned.
        # The latter distinction is important for complex NaNs, which are defined as at least one
        # of the real or imaginary parts being a NaN. The net effect is that NaNs are propagated.
        min_sigma = 1e-6
        N_normal = np.random.normal(0, 1, size=n)
        sigma_new = np.maximum(sigma_new, min_sigma)
        theta_new = theta + sigma_new * N_normal
        theta_new = np.clip(theta_new, -5, 5)
        return np.concatenate([theta_new, sigma_new])

    def perform_local_discrete(self, parents):
        "Creates one child"
        num_parents = len(parents)
        length = len(parents[0])
        child = np.zeros(length)
        for i in range(length):
            parent_idx = np.random.randint(0, num_parents)
            child[i] = parents[parent_idx][i]
        return child

    def fit(self, X_train, y_train, X_test=None, y_test=None):
        d = len(X_train[0])
        pop = self.initialize_population(d=d)
        for generation in range(self.max_generations):
            lambda_size = 7 * self.mu
            _lambda = []

            # X-Over
            for i in range(lambda_size):
                parent_indices = np.random.choice(
                    len(pop), size=2, replace=True)
                parents = [pop[idx] for idx in parent_indices]
                child = self.perform_local_discrete(parents)
                _lambda.append(child)

            # Mutate
            for i in range(len(_lambda)):
                _lambda[i] = self.mutate(_lambda[i], d + 1)

            fitness = []
            # Survivor Selection
            for ind in _lambda:
                theta = ind[:(d+1)]
                f = self.fitness(X_train, y_train, theta)
                fitness.append((f, ind))

            fitness.sort(key=lambda x: x[0], reverse=True)
            new_pop = [ind for (f, ind) in fitness[:self.mu]]
            pop = new_pop

            ###################################################################
            # Rest is for Analysis
            ###################################################################
            # Track best and mean fitness
            best_fitness = fitness[0][0]
            mean_fitness = np.mean([f for (f, ind) in fitness[:self.mu]])
            self.best_fitness_history.append(best_fitness)
            self.mean_fitness_history.append(mean_fitness)

            # Track accuracy
            best_theta = pop[0][:(d+1)]
            train_preds = self.predict(X_train, best_theta)
            train_accuracy = accuracy_score(y_train, train_preds)
            self.train_accuracy_history.append(train_accuracy)

            if X_test is not None and y_test is not None:
                test_preds = self.predict(X_test, best_theta)
                test_acc = accuracy_score(y_test, test_preds)
                self.test_accuracy_history.append(test_acc)

            # Print progress
            if (generation + 1) % 10 == 0 or generation == 0:
                print(f"Gen {generation+1:3d}/{self.max_generations} | "
                      f"Best Loss: {best_fitness:.4f} | "
                      f"Train Acc: {train_accuracy:.4f}")

        # Finalize: store best theta
        self.best_theta = pop[0][:(d+1)]

        return self

    def evaluate(self, X_test, y_test):
        y_pred = self.predict(X_test, self.best_theta)

        accuracy = accuracy_score(y_test, y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_test, y_pred, average='binary'
        )
        cm = confusion_matrix(y_test, y_pred)

        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'confusion_matrix': cm
        }

    def plot_results(self, save_prefix='es_results'):
        # Plot 1: Training Loss
        plt.figure(figsize=(10, 5))
        generations = range(1, len(self.best_fitness_history) + 1)

        plt.plot(generations, self.best_fitness_history,
                 label='Best Training Loss', color='blue')
        plt.plot(generations, self.mean_fitness_history,
                 label='Mean Training Loss', color='orange', alpha=0.7)

        plt.xlabel('Generation')
        plt.ylabel('Loss')
        plt.title('Training Loss vs Generation')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f'{save_prefix}_fitness.png', dpi=300)
        plt.show()

        # Plot 2: Accuracy
        plt.figure(figsize=(10, 5))

        plt.plot(generations, self.train_accuracy_history,
                 label='Training Accuracy', color='blue')

        if self.test_accuracy_history:
            plt.plot(generations, self.test_accuracy_history,
                     label='Test Accuracy', color='red')

        plt.xlabel('Generation')
        plt.ylabel('Accuracy')
        plt.title('Accuracy vs Generation')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f'{save_prefix}_accuracy.png', dpi=300)
        plt.show()

    def plot_confusion_matrix(self, cm, save_prefix='es_results'):
        plt.figure(figsize=(8, 6))

        # Display as heatmap
        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title('Confusion Matrix')
        plt.colorbar()

        # Add labels
        classes = ['No Disease (0)', 'Disease (1)']
        tick_marks = [0, 1]
        plt.xticks(tick_marks, classes)
        plt.yticks(tick_marks, classes)

        # Add text annotations
        thresh = cm.max() / 2
        for i in range(2):
            for j in range(2):
                plt.text(j, i, format(cm[i, j], 'd'),
                         ha='center', va='center',
                         color='white' if cm[i, j] > thresh else 'black',
                         fontsize=20)

        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.tight_layout()
        plt.savefig(f'{save_prefix}_confusion_matrix.png', dpi=300)
        plt.show()


def load_and_preprocess_data(csv_data):
    df = pd.read_csv(StringIO(csv_data))
    # .iloc[] is primarily integer position based (from 0 to length-1 of the axis), but may also be used with a boolean array.
    # Separate features (all columns except last) and target (last column)
    X = df.iloc[:, :-1].values
    target_col = df.iloc[:, -1].values

    # Binarize target:  0 -> 0 , else n -> 1
    target_col = (target_col > 0).astype(int)

    # Train-test split (70-30, stratified)
    # If you omit stratify,
    # rare classes may be underrepresented or missing in the test set.
    # sklearn.model_selection.train_test_split(*arrays, test_size=None, train_size=None, random_state=None, shuffle=True, stratify=None)
    # If None, the value is set to the complement of the train size. If train_size is also None, it will be set to 0.25.
    # Read More: https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html
    X_train, X_test, y_train, y_test = train_test_split(
        X, target_col, test_size=0.3, random_state=42, stratify=target_col
    )

    # Feature scaling (z-score standardization)
    mean = X_train.mean(axis=0)
    std = X_train.std(axis=0)

    # Avoid division by zero
    std[std == 0] = 1

    # Apply scaling
    X_train_scaled = (X_train - mean) / std
    X_test_scaled = (X_test - mean) / std

    return X_train_scaled, X_test_scaled, y_train, y_test


def load_csv_file(script_dir):

    possible_names = [
        'Heart_Disease_dataset.csv',
        'Heart Disease dataset.csv',
        'heart_disease_dataset.csv',
        'Heart_Disease_Dataset.csv',
        'HeartDiseaseDataset.csv',
        'heart.csv',
        'heart-disease.csv'
    ]

    # Directories to search
    possible_dirs = [
        script_dir,                              # Same directory as script
        os.path.join(script_dir, 'Extras'),      # Extras subdirectory
        os.path.join(script_dir, 'extras'),      # lowercase version
        os.path.join(script_dir, 'data'),        # common data folder name
        os.path.join(script_dir, 'Data'),
    ]

    for directory in possible_dirs:
        for filename in possible_names:
            file_path = os.path.join(directory, filename)
            if os.path.exists(file_path):
                print(f"[SUCCESS] Found CSV file: {file_path}")
                try:
                    with open(file_path, 'r') as f:
                        csv_data = f.read()
                    print(
                        f"[SUCCESS] Loaded CSV data ({len(csv_data):,} characters)")
                    return csv_data
                except Exception as e:
                    print(f"[ERROR] Could not read file: {e}")

    # Print what was searched if not found
    print("[NOT FOUND] Searched in:")
    for directory in possible_dirs:
        print(f"  - {directory}")

    return None


def main():

    print("=" * 60)
    print("Evolution Strategies for Heart Disease Classification")
    print("=" * 60)
    print()

    # Get the directory where the script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    csv_data = load_csv_file(script_dir)

    if csv_data is None:
        print()
        print("=" * 60)
        print("[ERROR] Could not find the CSV file!")
        print()
        print("Please make sure 'Heart_Disease_dataset.csv' is in the same")
        print("folder as this Python script.")
        print()
        print("Expected location:", os.path.join(
            script_dir, 'Heart_Disease_dataset.csv'))
        return

    print()
    print("=" * 60)
    print()
    print("Hyperparameters:")
    print("  - mu (parents): 30")
    print("  - lambda (offspring): 210")
    print("  - lambda_reg (L2 regularization): 0.01")
    print("  - max_generations: 100")
    print("  - Selection: (mu , lambda)-ES")
    print()

    # Load and preprocess data
    X_train, X_test, y_train, y_test = load_and_preprocess_data(csv_data)

    print(f"Training set size: {X_train.shape[0]}")
    print(f"Test set size: {X_test.shape[0]}")
    print(f"Number of features: {X_train.shape[1]}")
    print(f"Class distribution (train): {np.bincount(y_train)}")
    print(f"Class distribution (test): {np.bincount(y_test)}")
    print()

    # Initialize and train ES
    es = EvolutionStrategyLogisticRegression(
        mu=30,
        lambda_offspring=210,
        lambda_reg=0.01,
        max_generations=100,
    )

    print("Training Evolution Strategy...")
    print()
    es.fit(X_train, y_train, X_test, y_test)

    # Evaluate on test set
    print()
    print("=" * 60)
    print("Final Results on Test Set")
    print("=" * 60)
    results = es.evaluate(X_test, y_test)

    print(f"Accuracy:  {results['accuracy']:.4f}")
    print(f"Precision: {results['precision']:.4f}")
    print(f"Recall:    {results['recall']:.4f}")
    print(f"F1-Score:  {results['f1_score']:.4f}")
    print()
    print("Confusion Matrix:")
    print(results['confusion_matrix'])
    print()

    # Generate plots
    print("Generating plots...")
    es.plot_results()
    es.plot_confusion_matrix(results['confusion_matrix'])
    print("Plots saved successfully!")


if __name__ == "__main__":
    main()
