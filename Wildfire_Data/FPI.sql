--wildfire.fpi definition

drop materialized view if exists wildfire.fpi;
create materialized view wildfire.fpi as
select distinct 
	lat as fpi_lon, -- NOTE THAT DATA WAS LOADED INCORRECTLY AND IS CORRECTED HERE
	lng as fpi_lat, -- NOTE THAT DATA WAS LOADED INCORRECTLY AND IS CORRECTED HERE
	round(lat::numeric, 1) as fpi_lon_rnd, -- NOTE THAT DATA WAS LOADED INCORRECTLY AND IS CORRECTED HERE
	round(lng::numeric, 1) as fpi_lat_rnd, -- NOTE THAT DATA WAS LOADED INCORRECTLY AND IS CORRECTED HERE
	fpi,
	first_date as fpi_date,
	year as fpi_year
from 
	staging.fpi fpi
where 
	fpi.year between 2009 and 2015
	and fpi.fpi between 0 and 100
	and fpi.lng between 37 and 42
	and fpi.lat between -124 and -120
;

CREATE INDEX fpi_fire_lat_idx ON wildfire.fpi (fpi_lat);
CREATE INDEX fpi_fire_lon_idx ON wildfire.fpi (fpi_lon);
CREATE INDEX fpi_fire_year_idx ON wildfire.fpi (fpi_year);
CREATE INDEX fpi_fire_date_idx ON wildfire.fpi (fpi_date);

GRANT SELECT ON TABLE wildfire.fpi TO students;

refresh materialized view wildfire.fpi;
