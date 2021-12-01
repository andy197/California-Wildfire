-- wildfire.fod defintion

drop materialized view if exists wildfire.fod;
create materialized view wildfire.fod as
select
	latitude as fire_lat, 
	longitude as fire_lon, 
	round(latitude::numeric, 1) as fire_lat_rnd, 
	round(longitude::numeric, 1) as fire_lon_rnd,
	state, 
	fire_year::integer as fire_year,
	(to_date(fire_year::varchar, 'YYYY') + make_interval(days := discovery_doy::integer))::date as fire_date,
	fod_id
	
from
	staging."FOD_Fires"
where
	state='CA'
	and fire_year between 2009 and 2015
	and latitude between 37 and 42 --San Francisco and North
	and longitude between -124 and -120 -- cost of California to the Vertical border in the North
;

CREATE INDEX fod_fire_lat_idx ON wildfire.fod (fire_lat);
CREATE INDEX fod_fire_lon_idx ON wildfire.fod (fire_lon);
CREATE INDEX fod_fire_year_idx ON wildfire.fod (fire_year);
CREATE INDEX fod_fire_date_idx ON wildfire.fod (fire_date);

GRANT SELECT ON TABLE wildfire.fod TO students;

refresh materialized view wildfire.fod;
