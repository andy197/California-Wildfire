from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
import numpy as np
import os
from joblib import dump, load
from sklearn.metrics import f1_score, confusion_matrix, roc_curve, auc, classification_report, recall_score, precision_recall_curve
import pandas as pd
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE


class BaseModel:
    """
    Base Class that all other models import from - allows for models to reuse similar code and define there own methods
    where they deviate using inheritance
    """
    class_weight = {0: 1, 1: 80}  # Weighting for data imbalance - common to lr and rf so setting in the base class

    def __init__(self, model_name: str, training_data: pd.DataFrame, testing_data: pd.DataFrame, features: list = None, exclude: list = None, y_col="fire_occurrence", save_model=True, read_model_from_disk=True):
        """

        :param model_name: The name to use for the model when saving things to disk
        :param training_data: The training data set
        :param testing_data: The testing data set
        :param features: List of column names to use as features in the model
        :param exclude: List of column names to exclude from features - outdated
        :param y_col: The column to be predicted
        :param save_model: Flag to save model to disk
        :param read_model_from_disk: Flag to read model from disk
        """

        # Determine the column to use for features and the predictor
        x_cols = [key for key in training_data.keys() if key != y_col]
        if features is not None:
            x_cols = features
        if exclude is not None:
            x_cols = [key for key in x_cols if key not in exclude]

        # Make these values available later
        self.x_cols = x_cols
        self.y_col = y_col

        # Reduce the training and testing data down to the relevant columns
        all_keys = x_cols[:]
        all_keys.append(y_col)
        self.training_data = training_data[all_keys]
        self.testing_data = testing_data[all_keys]

        # Save other inputs as attributes to be available in other functions
        self.model_name = model_name
        self.save_model = save_model
        self.read_model_from_disk = read_model_from_disk
        self.data_path = os.path.dirname(__file__)
        self.model_path = os.path.join(self.data_path, model_name+".joblib")

    def train_model(self):
        """
        Handles the training of the model - determines if model should be read from disk and/or saved
        :return:
        """

        # Determine if model should be created or read
        if self.read_model_from_disk and os.path.exists(self.model_path):
            print("Reading model from disk {}".format(self.model_path))
            self.model = load(self.model_path)
        elif self.read_model_from_disk and not os.path.exists(self.model_path):
            print("Tried to read model from disk {}, but it did not exist - retraining".format(self.model_path))
            self.__train_model()
        elif not self.read_model_from_disk:
            print("Creating new model")
            self.__train_model()

        # Save model if desired
        if self.save_model:
            print("Saving model to disk {}".format(self.model_path))
            dump(self.model, self.model_path)

    def __train_model(self):
        """
        Defines the general process to train the model - each separate step is its own method call so that the
        inheriting classes can easily control how these steps are performed as needed
        :return:
        """
        # Data Cleanup - at a minimum it will drop nans
        clean_data = self.data_clean(self.training_data)

        # Get features and dependent feature data sets
        X_train, y_train = self.data_split(clean_data)

        # Get the model - initialized the model and must always be defined in the inheriting class
        self.model = self.get_model()

        # Train the model
        self.fit_model(X_train, y_train)

    def fit_model(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def data_clean(self, data):
        """
        Perform any cleaning on the data before the data is split into x and y
        :param data:
        :return:
        """
        data = data.dropna()

        return data

    def get_model(self):
        raise Exception("User must define how to get model from inheriting class")

    def data_split(self, training_data):
        X_train = training_data[self.x_cols]
        y_train = training_data[self.y_col]

        return X_train, y_train

    def model_stats(self, print_stats, save_stats):
        """
        Generates the model statistics to be used in the report and/or evaluation of model performance
        :param print_stats:
        :param save_stats:
        :return:
        """
        if print_stats:
            print("Statistics for {}".format(self.model_name))

        # Confusion Matrix
        cm = pd.DataFrame(confusion_matrix(self.y_test, self.y_pred))
        if print_stats:
            print("Confusion Matrix")
            print(cm)
            print("\n\n")
        if save_stats:
            cm.to_csv(os.path.join(self.data_path, self.model_name+"_CM.csv"))

        # Classification Report
        cr = classification_report(self.y_test, self.y_pred)
        if print_stats:
            print("Classification Report")
            print(cr)
            print("\n\n")
        if save_stats:
            with open(os.path.join(self.data_path, self.model_name + "_CR.txt"), "w") as f:
                f.write(cr)

        # Feature Importance
        features = self.x_cols
        importances = self.get_importances()

        feature_importances = [(feature, round(importance, 2)) for feature, importance in
                               zip(features, importances)]
        feature_importances = sorted(feature_importances, key=lambda x: x[1], reverse=True)
        print(self.model_name)
        if print_stats:
            [print('Variable: {:20} Importance: {}'.format(*pair)) for pair in feature_importances]
        if save_stats:
            plt.close("all")
            x_values = list(range(len(importances)))
            plt.bar(x_values, importances, orientation='vertical')
            plt.xticks(x_values, features, rotation='vertical')
            plt.ylabel('Importance')
            plt.xlabel('Variable')
            plt.title('Variable Importances')
            fig = plt.gcf()
            fig.set_size_inches(20, 15)
            plt.savefig(os.path.join(self.data_path, self.model_name+"_FI.png"))

        # Allow for unique stats for each model if needed
        self.extra_stats(print_stats, save_stats)

    def get_importances(self):
        return self.model.feature_importances_

    def extra_stats(self, print_stats, save_stats):
        """
        Doesn't need to do anything if the particular models doesn't have some specific measures that aren't already in
        model_stats()
        :param print_stats:
        :param save_stats:
        :return:
        """
        pass
    
    def predict(self):
        """
        Handles the process to generate the prediction data to be used in the model_stats calculations
        :return:
        """
        clean_data = self.data_clean(self.testing_data)
        X_test, self.y_test = self.data_split(clean_data)

        self.y_pred = self.model.predict(X_test)
        

class LogisticRegressionModel(BaseModel):
    """
    Logistic Regression Model from Wenjiao's Notebook
    """
    def __init__(self, weight, *args, **kwargs):
        """
        Extend the instantiation of this class to allow for a model weight flag to be set, pass the remaining args
        into the base class
        :param weight: Flags on whether to set the class_weight flag when creating the Model
        :param args:
        :param kwargs:
        """
        self.weight = weight
        super().__init__(*args, **kwargs)

    def get_model(self):
        """
        Defines how the model should be retrieved
        :return:
        """
        if self.weight:
            model = LogisticRegression(class_weight=self.class_weight)
        else:
            model = LogisticRegression()

        return model

    def get_importances(self):
        """
        Logistic Regression has its own way to get feature importances
        :return:
        """
        return self.model.coef_[0]


class RandomForestModel(BaseModel):
    """
    Random Forest Model from Dami's Notebook
    """
    def __init__(self, weight, *args, **kwargs):
        """
        Extend the instantiation of this class to allow for a model weight flag to be set, pass the remaining args
        into the base class
        :param weight: Flags on whether to set the class_weight flag when creating the Model
        :param args:
        :param kwargs:
        """
        self.weight = weight
        super().__init__(*args, **kwargs)

    def get_model(self):
        """
        Defines how the model should be retrieved
        :return:
        """
        RSEED = 50
        if self.weight:
            model = RandomForestClassifier(n_estimators=10, random_state=RSEED, bootstrap=True, class_weight=self.class_weight)
        else:
            model = RandomForestClassifier(n_estimators=10, random_state=RSEED, bootstrap=True)
        return model


class NeuralNetModel(BaseModel):
    """
    Logistic Regression Model from Nikki's Notebook
    """
    def data_clean(self, data, train=True):
        """
        Overrides how the data should be cleaned based on the process Nikki went through
        :param data:
        :return:
        """
        for col in self.x_cols:
            data[col] = pd.to_numeric(data[col], errors="coerce")
        data[self.y_col] = pd.to_numeric(data[self.y_col], errors="coerce")

        data = data.dropna()

        indices_to_keep = ~data.isin([np.nan, np.inf, -np.inf]).any(1)
        data = data[indices_to_keep].astype(np.float64)

        if train:
            # class count
            count_class_0, count_class_1 = data.fire_occurrence.value_counts()

            # divide by class
            df_class_0 = data[data['fire_occurrence'] == 0]
            df_class_1 = data[data['fire_occurrence'] == 1]

            df_class_1_over = df_class_1.sample(count_class_0, replace=True)
            df_test_over = pd.concat([df_class_0, df_class_1_over], axis=0)

            return df_test_over
        else:
            return data

    def get_model(self):
        """
        Neural Net Model from Nikki's Notebook
        """
        LRI = 0.0001
        rs = 27
        model = MLPClassifier(hidden_layer_sizes=(30, 30, 2), max_iter=500, random_state=rs,
                              learning_rate_init=LRI)

        return model

    def predict(self):
        """
        Handles the process to generate the prediction data to be used in the model_stats calculations
        :return:
        """
        clean_data = self.data_clean(self.testing_data, train=False)
        X_test, self.y_test = self.data_split(clean_data)

        self.y_pred = self.model.predict(X_test)

    def get_importances(self):
        # No concept of feature importance for MLP Classifier
        return [0 for i in range(len(self.x_cols))]


class GradientBoostingModel(BaseModel):
    """
    Will be the Ensemble Model from Wenjiao's Notebook after I confirm what I should use
    """

    def data_clean(self, data):
        """
        Perform any cleaning on the data before the data is split into x and y
        :param data:
        :return:
        """
        data = data.fillna(value=-1)

        return data

    def data_split(self, training_data, train=True):
        X_train = training_data[self.x_cols]
        y_train = training_data[self.y_col]

        if train:
            # UNIQUE SMOTE STEP
            sm = SMOTE(random_state=10)
            X_train_wsm, y_train_wsm = sm.fit_sample(X_train, y_train)

            return X_train_wsm, y_train_wsm
        else:
            return X_train, y_train

    def get_model(self):
        model = GradientBoostingClassifier(n_estimators=30)
        return model

    def predict(self):
        """
        Handles the process to generate the prediction data to be used in the model_stats calculations
        :return:
        """
        clean_data = self.data_clean(self.testing_data )
        X_test, self.y_test = self.data_split(clean_data, train=False)

        self.y_pred = self.model.predict(X_test)