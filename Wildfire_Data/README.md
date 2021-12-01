# COMBINED
The primary data feed. Joins the two tables below on their rounded latitude and longitude columns and date. See column defintion from other two tables for why and how the columns have been rounded. Conner can also probably explain it better directly. Also see the other table explanations for what subset of data is represented in this table.

Many of the fields in this table are arrays - due to the nature of the join, multiple FPIs and fire occurrences can be rolled up. These arrays seek to make these roll-ups transparent so that they can be accounted for and addressed if needed by the Algorithm team.

**NOTE** The order of the values in the different arrays appears to random, but I think could be ordered if there is value in that. Just ask Conner to dig into if needed.

An explanation of the columns are as follows:
- **lat_rnd:** The rounded latitude that joins the FPI and FOD data
- **lon_rnd:** The rounded longitude that joins the FPI and FOD data
- **fpi_date:** The date that joins the FPI and FOD data (fpi calculation date and or fire occurrence date)
- **fire_occurrence:** A flag, either 1 or 0, identifying if any of the rolled up FOD_IDs were not null meaning fire did occur on that date in the rounded location (1 - fire, 0 - no fire)
- **max_fpi:** The maximum fpi value for the group
- **min_fpi:** The minimum fpi value for the group
- **avg_fpi:** The average fpi value for the group
- **array_fpi:** All fpi values for the group
- **array_fod_lat:** All raw Fire Occurrence latitudes for the group
- **array_fod_lon:** All raw Fire Occurrence longitudes for the group
- **array_fpi_lat:** All raw FPI latitudes for the group
- **array_fpi_lon:** All raw FPI longitues for the group
- **array_fpi_fod_id:** All Fire Occurrences for the group
- **rolled_up_rows:** A count of the number of rows for the group

## FOD
From the Fire Occurrence Database. Narrowed down to an area in the State of California, stretching from San Francisco North 
(latitude from 37 to 43) and from the coast to the eastern-vertical boder (longitude from -124 to -120). The years of data range from 2009 to 2015.

The records in this data denote the occurrence of a fire for a given location and date.

An explanation of the columns are as follows:
- **fire_lat:** The latitude column from the FOD => the point location of the fire
- **fire_lon:** The longitude column from the FOD => the point location of the fire
- **fire_lat_rnd:** The latitude column from the FOD rounded to 1 decimal point (this resolution resluted in the greatest number of "exact" joins between the FOD data and the FPI data)
- **fire_lon_rnd:** The longitude column from the FOD rounded to 1 decimal point (this resolution resluted in the greatest number of "exact" joins between the FOD data and the FPI data)
- **state:** The state column from the FOD where the fire occurred => Two letter state code
- **fire_year:** The fire_year column from the FOD defining the year when the fire occurred
- **fire_date:** The fire_year column and the discovery_doy from the FOD summed together to get the fire occurrence date
- **fod_id:** The fod_id colum from the FOD => unique id of the fire

The definition for each FOD column can be found here: https://www.fs.usda.gov/rds/archive/products/RDS-2013-0009.4/_metadata_RDS-2013-0009.4.html

## FPI
From the FPI Images that Conner parsed and loaded using the FPI_LOAD scripts. Narrowed down to an area in the State of California, stretching from San Francisco North (latitude from 37 to 43) and from the coast to the eastern-vertical boder (longitude from -124 to -120). The years of data range from 2009 to 2015. The original data was calculated to 1 Sqr Mile, and Conner loaded every 5-th data point to save on processing time (this can be reloaded to better resolution if needed). The FPI values have been limited from 0-100 (values outside of this range represent water and other non-flamable features. See the FPI_LOAD script for a rough explanation of what the other values could represent. 

The records in this data denote daily FPI value for a given location and date.

An explanation of the columns are as follows:
- **fpi_lat:** The latitude of the centroid of the 1 Sqr Mile. area of the FPI value
- **fire_lon:** The longide of the centroid of the 1 Sqr Mile. area of the FPI value
- **fire_lat_rnd:** The latitude column rounded to 1 decimal point (this resolution resluted in the greatest number of "exact" joins between the FOD data and the FPI data)
- **fire_lon_rnd:** The longitude column rounded to 1 decimal point (this resolution resluted in the greatest number of "exact" joins between the FOD data and the FPI data)
- **fpi:** The daily FPI values for the 1 Sqr. Mile area.
- **fpi_date:** The first_date column from Conner's processing script - derived from the file name. Represents the data of the FPI calculation
- **fpi_year:** The year from the download request. Represents the year of the FPI calculation.
