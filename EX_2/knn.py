
import numpy as np
import faiss
from helpers import *
import pandas as pd


class KNNClassifier:
    def __init__(self, k, distance_metric='l2'):
        self.k = k
        self.distance_metric = distance_metric
        self.X_train = None
        self.Y_train = None

    def fit(self, X_train, Y_train):
        """
        Update the kNN classifier with the provided training data.

        Parameters:
        - X_train (numpy array) of size (N, d): Training feature vectors.
        - Y_train (numpy array) of size (N,): Corresponding class labels.
        """
        self.X_train = X_train.astype(np.float32)
        self.Y_train = Y_train
        d = self.X_train.shape[1]
        if self.distance_metric == 'l2':
            self.index = faiss.index_factory(d, "Flat", faiss.METRIC_L2)
        elif self.distance_metric == 'l1':
            self.index = faiss.index_factory(d, "Flat", faiss.METRIC_L1)
        else:
            raise NotImplementedError
        pass
        self.index.add(self.X_train)

    @staticmethod
    def mode(array):
        unique, counts = np.unique(array, return_counts=True)
        max_count_index = np.argmax(counts)
        return unique[max_count_index]

    def predict(self, X):
        """
        Predict the class labels for the given data.

        Parameters:
        - X (numpy array) of size (M, d): Feature vectors.

        Returns:
        - (numpy array) of size (M,): Predicted class labels.

        distance, test_features_faiss_Index = self.index.search(test_features, self.k) 
        self.test_label_faiss_output = stats.mode(self.train_labels[test_features_faiss_Index],axis=1)[0]
        self.test_label_faiss_output = np.array(self.test_label_faiss_output.ravel())
        return self.test_label_faiss_output
        """
        distances, indexs = self.knn_distance(X)  # use knn
        labels = []
        for i in range(len(indexs)):
            labels.append(KNNClassifier.mode(self.Y_train[indexs[i]]))
        return np.array(labels)

    def knn_distance(self, X):
        """
        Calculate kNN distances for the given data. You must use the faiss library to compute the distances.
        See lecture slides and https://github.com/facebookresearch/faiss/wiki/Getting-started#in-python-2 for more information.

        Parameters:
        - X (numpy array) of size (M, d): Feature vectors.

        Returns:
        - (numpy array) of size (M, k): kNN distances.
        - (numpy array) of size (M, k): Indices of kNNs.
        """
        m, d = X.shape
        X = X.astype(np.float32)
        distances, indexs = self.index.search(X, self.k)
        return distances, indexs


def scenario1():
    df_train = pd.read_csv("train.csv")
    df_test = pd.read_csv("test.csv")
    X_train, X_test, Y_train, Y_test = df_train.drop(columns=["state"]), df_test.drop(
        columns=["state"]), df_train["state"], df_test["state"]
    result = []
    for k in {1, 10, 100, 1000, 3000}:
        for l in {"l1", "l2"}:
            knn_classifier = KNNClassifier(k=k, distance_metric=l)
            knn_classifier.fit(X_train, Y_train)
            y_pred = knn_classifier.predict(X_test)
            accuracy = np.mean(y_pred == Y_test)
            result.append({"k": k, "l": l, "accuracy": accuracy})
    df = pd.DataFrame(result)
    print(df)


def scenario2():
    df_train = pd.read_csv("train.csv")
    df_test = pd.read_csv("test.csv")
    X_train, X_test, Y_train, Y_test = df_train.drop(columns=["state"]), df_test.drop(
        columns=["state"]), df_train["state"], df_test["state"]
    result = []
    try:
        k = int(input("k:"))
    except ValueError:
        print("k must be integer")
        return
    for l in {"l1", "l2"}:
        knn_classifier = KNNClassifier(k=k, distance_metric=l)
        knn_classifier.fit(X_train, Y_train)
        y_pred = knn_classifier.predict(X_test)
        accuracy = np.mean(y_pred == Y_test)
        result.append({"k": k, "l": l, "accuracy": accuracy})
    df = pd.DataFrame(result)
    print(df)


def scenario3():
    K_max = 1
    # 0.966711
    K_min = 3000
    # 0.398136
    df_train = pd.read_csv("train.csv")
    df_test = pd.read_csv("test.csv")
    X_train = df_train.drop(columns=["state"]).values
    Y_train = df_train["state"].values
    X_test = df_test.drop(columns=["state"]).values
    Y_test = df_test["state"].values
    knn_classifier = KNNClassifier(k=K_max, distance_metric="l2")
    knn_classifier.fit(X_train, Y_train)
    y_pred = knn_classifier.predict(X_test)
    plot_decision_boundaries(knn_classifier, X_test, Y_test, "Kmax and L2")
    plt.show()


def scenario4():
    K_min = 3000
    # 0.398136
    df_train = pd.read_csv("train.csv")
    df_test = pd.read_csv("test.csv")
    X_train = df_train.drop(columns=["state"]).values
    Y_train = df_train["state"].values
    X_test = df_test.drop(columns=["state"]).values
    Y_test = df_test["state"].values
    knn_classifier = KNNClassifier(k=K_min, distance_metric="l2")
    knn_classifier.fit(X_train, Y_train)
    y_pred = knn_classifier.predict(X_test)
    plot_decision_boundaries(knn_classifier, X_test, Y_test, "Kmin and L2")
    plt.show()


def scenario5():
    K_max = 1
    df_train = pd.read_csv("train.csv")
    df_test = pd.read_csv("test.csv")
    X_train = df_train.drop(columns=["state"]).values
    Y_train = df_train["state"].values
    X_test = df_test.drop(columns=["state"]).values
    Y_test = df_test["state"].values
    knn_classifier = KNNClassifier(k=K_max, distance_metric="l1")
    knn_classifier.fit(X_train, Y_train)
    y_pred = knn_classifier.predict(X_test)
    plot_decision_boundaries(knn_classifier, X_test, Y_test, "Kmax and L1")
    plt.show()


def scenario6():
    knn_classifier = KNNClassifier(k=5, distance_metric='l2')
    df_train = pd.read_csv("train.csv")
    df_test = pd.read_csv("AD_test.csv")
    X_test = df_test.values
    X_train = df_train.drop(columns=["state"]).values
    Y_train = df_train["state"].values
    knn_classifier.fit(X_train, Y_train)
    index = faiss.index_factory(X_train.shape[1], "Flat", faiss.METRIC_L2)
    index.add(X_train.astype(np.float32))
    distances, indices = index.search(df_test, 5)

    # get anomaly scores
    anomaly_scores = np.sum(distances, axis=1)
    top_50_anomalies_indexs = np.argsort(anomaly_scores)[-50:]
    anomalies_x = X_test[top_50_anomalies_indexs, 0]
    anomalies_y = X_test[top_50_anomalies_indexs, 1]
    plt.figure(figsize=(10, 10))
    plt.scatter(X_test[:, 0], X_test[:, 1], color='blue',
                label='Test', alpha=0.5, s=3)
    plt.scatter(anomalies_x, anomalies_y, color='red',
                label='Anomaly', alpha=1.0, s=3)
    plt.scatter(X_train[:, 0], X_train[:, 1], color='black',
                label='Normal', alpha=0.01, s=3)

    plt.xlabel('LONG')
    plt.ylabel('LAT')
    plt.title('Scatter Plot of test and train data')
    plt.legend()
    plt.show()
def train_decision_tree(X_train, Y_train, X_test, Y_test,X_val,Y_val):
    max_depths = [1,2,4,6,10,20,50,100]
    max_leaf_nodes = [50,100,1000]
    trained_model_trees= []

    for max_depth in max_depths:
        for max_leaf_node in max_leaf_nodes:
            tree_classifier=DecisionTreeClassifier(max_depth=max_depth,max_leaf_nodes=max_leaf_node)
            tree_classifier.fit(X_train,Y_train)

            train_accuracy = accuracy_score(Y_train,tree_classifier.predict(X_train))

            val_accuracy = accuracy_score(Y_val,tree_classifier.predict(X_val))

            test_accuracy = accuracy_score(Y_test,tree_classifier.predict(X_test))

            model_info ={
                'model': tree_classifier,
                'hyperparameters': {'max_depth': max_depth, 'max_leaf_nodes': max_leaf_node},
                'accuracy': {'train': train_accuracy, 'validation': val_accuracy, 'test': test_accuracy}
            }
            trained_model_trees.append(model_info)
    return trained_model_trees
def scenario7b():
    data_numpy, col_names = read_data_demo(filename='train.csv')
    x = data_numpy[:,:-1] 
    y = data_numpy[:,-1]
    X_train, X_temp, y_train, y_temp = train_test_split(x, y, test_size=0.4, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)
    trained_models = train_decision_tree(X_train, y_train, X_val, y_val, X_test, y_test)
    print("Decision Tree Results")
    for model in trained_models:
        print("Hyperparameters: ", model['hyperparameters'])
        print("Accuracy: ", model['accuracy'])
        title = f"Decision Tree Model\nMax Leaves: {model['hyperparameters']['max_leaf_nodes']}, Max Depth: {model['hyperparameters']['max_depth']}\nTrain data"
        plot_decision_boundaries( model['model'],X_train, y_train,
                                 title)
        print()
def scenario7():
    data_numpy, col_names = read_data_demo(filename='train.csv')
    x = data_numpy[:,:-1] 
    y = data_numpy[:,-1]
    X_train, X_temp, y_train, y_temp = train_test_split(x, y, test_size=0.4, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

    trained_models = train_decision_tree(X_train, y_train, X_val, y_val, X_test, y_test)
    print("Decision Tree Results")
    for model in trained_models:
        print("Hyperparameters: ", model['hyperparameters'])
        print("Accuracy: ", model['accuracy'])
        plot_decision_boundaries(model['model'], X_test, y_test, "Decision Tree")
        plt.show()
def random_forest_demo(X_train, y_train, X_val, y_val, X_test, y_test):
    rf_classifier = RandomForestClassifier(n_estimators=300, max_depth=6, random_state=42)

    rf_classifier.fit(X_train, y_train)

    y_val_pred = rf_classifier.predict(X_val)
    y_test_pred = rf_classifier.predict(X_test)

    val_accuracy = accuracy_score(y_val, y_val_pred)
    test_accuracy = accuracy_score(y_test, y_test_pred)

    print("Validation accuracy: ", val_accuracy)
    print("Test accuracy: ", test_accuracy)

    return {
        'model': rf_classifier,
        'hyperparameters': {
            'n_estimators': 300,
            'max_depth': 6,
        },
        'accuracy': {
            'validation': val_accuracy,
            'test': test_accuracy,
        },
    }
def scenario8():
    data_numpy, col_names = read_data_demo(filename='train.csv')
    x = data_numpy[:,:-1] 
    y = data_numpy[:,-1]
    X_train, X_temp, y_train, y_temp = train_test_split(x, y, test_size=0.4, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)
    trained_model = random_forest_demo(X_train, y_train, X_val, y_val, X_test, y_test)
    print("Random Forest Results")
    print("Hyperparameters: ", trained_model['hyperparameters'])
    print("Accuracy: ", trained_model['accuracy'])
    #show the results:
    plot_decision_boundaries(trained_model['model'], X_test, y_test, "Random Forest")
    return
# def xgboost_demo(X_train,y_train,X_val,y_val,X_test,y_test):
#     xgb_classifier =loading_xgboost()
#     xgb_classifier.fit(X_train,y_train)

#     y_val_pred = xgb_classifier.predict(X_val)
#     y_test_pred = xgb_classifier.predict(X_test)

#     val_accuracy = accuracy_score(y_val, y_val_pred)
#     test_accuracy = accuracy_score(y_test, y_test_pred)

#     print("Validation accuracy: ", val_accuracy)
#     print("Test accuracy: ", test_accuracy)

#     return {
#         'model': xgb_classifier,
#         'hyperparameters': {
#             'n_estimators': 300,
#             'max_depth': 6,
#             'learning_rate': 0.1,
#         },
#         'accuracy': {
#             'validation': val_accuracy,
#             'test': test_accuracy,
#         },
#     }
# def scenario9():
#     data_numpy, col_names = read_data_demo(filename='train.csv')
#     x = data_numpy[:,:-1] 
#     y = data_numpy[:,-1]
#     X_train, X_temp, y_train, y_temp = train_test_split(x, y, test_size=0.4, random_state=42)
#     X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)
#     trained_model = xgboost_demo(X_train, y_train, X_val, y_val, X_test, y_test)
#     print("XGBoost Results")
#     print("Hyperparameters: ", trained_model['hyperparameters'])
#     print("Accuracy: ", trained_model['accuracy'])
#     importance = trained_model['model'].feature_importances_
    
#     for i,v in enumerate(importance):
#         print('Feature: %0d, Score: %.5f' % (i,v))

#     plt.bar([x for x in range(len(importance))], importance)
#     plt.savefig('xgboost_feature_importance.png')
#     plt.show()
if __name__ == '__main__':
    np.random.seed(0)
    # scenario1()
    # scenario2()
    # scenario3()
    # scenario4()
    # scenario5()
    # scenario6()
    # scenario7()
    scenario8()
    # scenario9()
    # scenario7b()