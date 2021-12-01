import os
from sqlalchemy import create_engine
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

class Data:
    """
    Class to handle the retrieval of data from the database for training, testing, and visualization (to-do)
    """
    def __init__(self, db_username: str, db_password: str, db_host: str, db_dbname: str):
        # Create a connection to the database where data has already been loaded
        self.engine = create_engine('postgresql://{username}:{password}@{host}/{dbname}'.format(
            username=db_username,
            password=db_password,
            host=db_host,
            dbname=db_dbname
        ))

        # Set the save director
        self.save_path = os.path.dirname(__file__)

    def pull_training_data(self, save_data=True, from_disk=True, subset_data=True) -> pd.DataFrame:
        """
        Reads in the training data use to generate the models
        :param save_data: Saves the data to disk to save time on getting the data
        :param from_disk: Reads the data from disk
        :param subset_data: Pull only data from July
        :return:
        """

        # Where the data will be saved to / read from
        data_path = os.path.join(self.save_path, "training_data.csv")

        # The sql query to be run
        if subset_data:
            sql = 'SELECT *, extract(month from fpi_date) as month FROM wildfire.combined WHERE extract(month from fpi_date)=7 and extract(year from fpi_date) < 2015 and extract(year from fpi_date) >= 2009'
        else:
            sql = 'SELECT *, extract(month from fpi_date) as month FROM wildfire.combined WHERE extract(year from fpi_date) < 2015 and extract(year from fpi_date) >= 2009'

        if from_disk and os.path.exists(data_path):
            print("Reading training data from {}".format(data_path))
            self.training_data = pd.read_csv(data_path)
        elif from_disk and not os.path.exists(data_path):
            print("Tried reading training data from {}, but it did not already exist - generating from the db".format(data_path))
            # Use the engine to connect to the db and pull all combined data from 2009-2014 for the month of July
            self.training_data = pd.read_sql(sql, self.engine)
        elif not from_disk:
            print("Generating training data from the db".format(data_path))
            # Use the engine to connect to the db and pull all combined data from 2009-2014 for the month of July
            self.training_data = pd.read_sql(sql, self.engine)

        print("Got the training data!")
        if save_data:
            print("Training data was requested to be saved")
            self.training_data.to_csv(data_path, index=False)
            print("Training data was saved")

        return self.training_data

    def pull_testing_data(self, save_data=True, from_disk=True, subset_data=True) -> pd.DataFrame:
        """
        Reads in the testing data use to evaluate the models
        :param save_data: Saves the data to disk to save time on getting the data
        :param from_disk: Reads the data from disk
        :param subset_data: Pull only data from July
        :return:
        """
        # Where the data will be saved to / read from
        data_path = os.path.join(self.save_path, "testing_data.csv")

        # The sql query to be run
        if subset_data:
            sql = 'SELECT *, extract(month from fpi_date) as month FROM wildfire.combined WHERE extract(month from fpi_date)=7 and extract(year from fpi_date)=2015'
        else:
            sql = 'SELECT *, extract(month from fpi_date) as month FROM wildfire.combined WHERE extract(year from fpi_date) = 2015'

        if from_disk and os.path.exists(data_path):
            print("Reading testing data from {}".format(data_path))
            self.testing_data = pd.read_csv(data_path)
        elif from_disk and not os.path.exists(data_path):
            print("Tried reading testing data from {}, but it did not already exist - generating from the db".format(
                data_path))
            # Use the engine to connect to the db and pull all combined data from 2009-2014 for the month of July
            self.testing_data = pd.read_sql(sql, self.engine)
        elif not from_disk:
            print("Generating testing data from the db".format(data_path))
            # Use the engine to connect to the db and pull all combined data from 2009-2014 for the month of July
            self.testing_data = pd.read_sql(sql, self.engine)

        print("Got the testing data!")
        if save_data:
            print("Testing data was requested to be saved")
            self.testing_data.to_csv(data_path, index=False)
            print("Testing data was saved")

        return self.testing_data

    def pull_visualization_data(self,  save_data=True, from_disk=True) -> pd.DataFrame:
        raise NotImplemented("Conner has not implemented this feature yet")

    def training_data_stats(self, print_results=True, save=True) -> None:
        """
        Generates various statistics about the training data set

        :param print_results: Prints the results to the console
        :param save: Save plots and tables to disk
        :return: None
        """
        # Set pandas to show every column
        pd.set_option('display.max_columns', None)

        # Describe the data
        desc = self.training_data.describe(include='all')
        if print_results:
            print("Data Stats")
            print(desc)
            print("End of Data Stats\n\n")
        if save:
            print("Saving data stats")
            desc.to_csv(os.path.join(self.save_path, "data_stats.csv"))

        # Get independent and dependent columns and datasets
        y_col = "fire_occurrence"
        x_cols = [key for key in self.training_data.keys() if key != y_col]

        X_train = self.training_data[x_cols]
        y_train = self.training_data[y_col]

        if print_results:
            print("Independent Parameters")
            print(x_cols)
            print("Dependent Parameters")
            print(y_col)
            print("\n\n")

        # Data Imbalance
        fire = y_train.isin([1]).sum()
        no_fire = y_train.isin([0]).sum()
        no_fireP = (no_fire / (fire + no_fire)) * 100
        fireP = (fire / (fire + no_fire)) * 100
        if print_results:
            print("No Fire %: " + str(no_fireP))
            print("Fire %: " + str(fireP))

        labels = ['No Fire', 'Fire']
        data = [no_fireP, fireP]
        plt.bar(labels, data)
        plt.ylabel("Count")
        plt.title("Fire Occurrence Imbalance")
        if save:
            plt.savefig(os.path.join(self.save_path, "fire_data_imbalance.png"))
        if print_results:
            plt.show()

        # Data Correlation
        cor = X_train.corr()
        plt.figure(figsize=(16, 10))
        plt.title("Feature Correlation")
        sns.heatmap(cor)
        if save:
            plt.savefig(os.path.join(self.save_path, "feature_data_correlation.png"))
        if print_results:
            plt.show()
