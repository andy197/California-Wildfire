# Guide to create input features (sattelite and map-based) to model LFMC

example usage:


	python download_landsat.py -p IDAHO_EPSCOR/TERRACLIMATE \
	                -b B1 B2 B3 B4 B5 B6 B7 pixel_qa  \
	                -s "2019-06-01" \
	                -e "2020-02-27" \
	                -f "nfmd_queried_cali.csv" \
	                -sc 500 \
	                -d "drought_Soil_moisture" \
	                -b pdsi soil

specify the bands on google earth engine. Some maps do not have bands, in that case omit -b.
see source for args.
Need to register on google earth engine in order to authenticate.
CSV needs to be in the format of

name,lat,long

Need to modify code if name is empty.

credits to Krishna Rao
https://github.com/kkraoj/lfmc_from_sar/tree/master/input_data_creation
