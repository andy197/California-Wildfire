# Configure the Code
import os

# Set your credentials to connect to the database
db_username = "username"
db_password = "pwd"
db_host = "host"
dbname = "dbname"

# Training/Testing data flags
save_data = False  # Sets whether the training data should be saved to disk
from_disk = True  # Sets where the training data should be read from disk if you had previously saved it
subset_data = False  # Pull only the data from July

# Data statistics
print_results = False  # Prints the data statistics to your console
save = False  # Saves the data statistics to disk inside the Data/ directory

# LR
lr_save_model = False  # Saves the Logistic Regression Models to disk - other models have similar flags
lr_read_model = True  # Reads the Logistic Regression Models from disk if you have previously saved it - other models have similar flags

# RF
rf_save_model = False
rf_read_model = True

# NN
nn_save_model = False
nn_read_model = True

# GB
gb_save_model = False
gb_read_model = True

# Model Stats
print_model_stats = True  # Print model statistics to console
save_model_stats = True  # Save model statistics to disk

# Import the main Packages
from Data.Data import Data
from Models.Models import RandomForestModel, LogisticRegressionModel, NeuralNetModel, GradientBoostingModel

if __name__ == "__main__":
    # Create the Object that will handle data retrieval (training and testing data sets)
    data = Data(db_username, db_password, db_host, dbname)

    # Pull training data - training data also save as attribute of class
    training_data = data.pull_training_data(save_data=save_data, from_disk=from_disk, subset_data=subset_data)

    # Data Stats - creates various descriptive statistics about the training data for use in the report
    if print_results or save:
        data.training_data_stats(print_results=print_results, save=save)

    # Pull testing data
    testing_data = data.pull_testing_data(save_data=save_data, from_disk=from_disk, subset_data=subset_data)

    # Train models
    # Set columns to exclude - date and arrays
    exclude = ["array_fpi", "array_fod_lat", "array_fod_lon", "array_fpi_lat", "array_fpi_lon", "array_fod_id",
               "fpi_date"]

    ### BEGIN MODEL TRAINING ###

    ## LOGISTIC REGRESSION ##
    # Train on the fpi data - Conner Added
    print("FPI LR")
    lr_fpi = LogisticRegressionModel(False, "FPI LR", training_data.copy(), testing_data.copy(),
                                     features=['max_fpi', 'min_fpi', 'avg_fpi'], save_model=lr_save_model,
                                     read_model_from_disk=lr_read_model)
    lr_fpi.train_model()

    # Train on all data - Wenjiao's method
    print("LR")
    lr = LogisticRegressionModel(False, "LR", training_data.copy(), testing_data.copy(),
                                 features=['lat_rnd', 'lon_rnd', 'max_fpi', 'min_fpi', 'avg_fpi'],
                                 save_model=lr_save_model, read_model_from_disk=lr_read_model)
    lr.train_model()

    # Train on the fpi data - weighted - Conner Added
    print("FPI WLR")
    wlr_fpi = LogisticRegressionModel(True, "FPI WLR", training_data.copy(), testing_data.copy(),
                                      features=['max_fpi', 'min_fpi', 'avg_fpi'], save_model=lr_save_model,
                                      read_model_from_disk=lr_read_model)
    wlr_fpi.train_model()

    # Train on all data - weighted - Wenjiao's method
    print("WLR")
    wlr = LogisticRegressionModel(True, "WLR", training_data.copy(), testing_data.copy(),
                                  features=['lat_rnd', 'lon_rnd', 'max_fpi', 'min_fpi', 'avg_fpi'],
                                  save_model=lr_save_model, read_model_from_disk=lr_read_model)
    wlr.train_model()

    ## RANDOM FOREST ## - Dami's method
    print("RF")
    rf = RandomForestModel(False, "RF", training_data.copy(), testing_data.copy(),
                           features=['lat_rnd', 'lon_rnd', 'max_fpi',
                                     'min_fpi', 'avg_fpi', 'std_fpi',
                                     'daily_ref_evapotrans_mm',
                                     'hundred_hour_dead_fuel_moist_percent',
                                     'precip_amount_mm',
                                     'max_relative_humidity_percent',
                                     'min_relative_humidity_percent',
                                     'specific_humidity_kg_kg', 'srad_wmm',
                                     'temp_min_k', 'temp_max_k',
                                     'mean_vapor_pressure_deficit_kpa',
                                     'wind_speed_10m_m_s', 'elevation', 'eng_release_comp_nfdrs'],
                           save_model=rf_save_model, read_model_from_disk=rf_read_model)
    rf.train_model()

    print("WRF")
    wrf = RandomForestModel(True, "WRF", training_data.copy(), testing_data.copy(),
                            features=['lat_rnd', 'lon_rnd', 'max_fpi',
                                      'min_fpi', 'avg_fpi', 'std_fpi',
                                      'daily_ref_evapotrans_mm',
                                      'hundred_hour_dead_fuel_moist_percent',
                                      'precip_amount_mm',
                                      'max_relative_humidity_percent',
                                      'min_relative_humidity_percent',
                                      'specific_humidity_kg_kg', 'srad_wmm',
                                      'temp_min_k', 'temp_max_k',
                                      'mean_vapor_pressure_deficit_kpa',
                                      'wind_speed_10m_m_s', 'elevation', 'eng_release_comp_nfdrs'],
                            save_model=rf_save_model, read_model_from_disk=rf_read_model)
    wrf.train_model()

    ## NEURAL NET ## - Nikki's method
    print("NN")
    nn = NeuralNetModel("NN", training_data.copy(), testing_data.copy(),
                        features=['lat_rnd', 'lon_rnd', 'elevation', 'max_fpi', 'min_fpi',
                                  'avg_fpi', 'std_fpi', 'eng_release_comp_nfdrs',
                                  'burning_index_nfdrs',
                                  'daily_ref_evapotrans_mm',
                                  'hundred_hour_dead_fuel_moist_percent',
                                  'precip_amount_mm',
                                  'max_relative_humidity_percent',
                                  'min_relative_humidity_percent',
                                  'specific_humidity_kg_kg',
                                  'srad_wmm', 'temp_min_k', 'temp_max_k',
                                  'mean_vapor_pressure_deficit_kpa', 'wind_speed_10m_m_s'], save_model=nn_save_model,
                        read_model_from_disk=nn_read_model)
    nn.train_model()

    ## GRADIENT BOOSTING  ## - Wenjiao's method
    print("GB")
    gb = GradientBoostingModel("GB", training_data.copy(), testing_data.copy(),
                               features=['lat_rnd', 'lon_rnd', 'max_fpi', 'min_fpi', 'avg_fpi', 'std_fpi', 'elevation',
                                         'daily_ref_evapotrans_mm',
                                         'hundred_hour_dead_fuel_moist_percent', 'precip_amount_mm',
                                         'eng_release_comp_nfdrs', 'burning_index_nfdrs',
                                         'max_relative_humidity_percent', 'min_relative_humidity_percent',
                                         'specific_humidity_kg_kg', 'srad_wmm', 'temp_min_k', 'temp_max_k',
                                         'mean_vapor_pressure_deficit_kpa', 'wind_speed_10m_m_s'],
                               save_model=gb_save_model,
                               read_model_from_disk=gb_read_model)
    gb.train_model()

    # Gather models to report statistics on
    models = [lr_fpi, lr, wlr_fpi, wlr, rf, wrf, nn, gb]

    # Model Stats
    for model in models:
        # Have the models predict on the testing set
        model.predict()

        # Generate model stats
        model.model_stats(print_stats=print_model_stats, save_stats=save_model_stats)