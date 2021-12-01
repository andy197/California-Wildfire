--wildfire.combined definition

drop materialized view if exists wildfire.combined cascade;
create materialized view wildfire.combined as
select 
	fpi_lat_rnd as lat_rnd,
	fpi_lon_rnd as lon_rnd,
	fpi_date,
	avg(e.elevation::numeric) AS elevation,
	max(case when fod_id is not null then 1 else 0 end) as fire_occurrence,
	max(fpi) as max_fpi,
	min(fpi) as min_fpi,
	avg(fpi) as avg_fpi,
	stddev(fpi) as std_fpi,
	array_agg(fpi) as array_fpi,
	array_agg(fire_lat) as array_fod_lat,
	array_agg(fire_lon) as array_fod_lon,
	array_agg(fpi_lat) as array_fpi_lat,
	array_agg(fpi_lon) as array_fpi_lon,
	array_agg(fod_id) as array_fod_id,
	count(*) as rolled_up_rows,
    avg(d2.erc) AS eng_release_comp_nfdrs,
    avg(d2.bi) AS burning_index_nfdrs,
    avg(d2.eto) AS daily_ref_evapotrans_mm,
    avg(d2.fm100) AS hundred_hour_dead_fuel_moist_percent,
    avg(d2.pr) AS precip_amount_mm,
    avg(d2.rmax) AS max_relative_humidity_percent,
    avg(d2.rmin) AS min_relative_humidity_percent,
    avg(d2.sph) AS specific_humidity_kg_kg,
    avg(d2.srad) AS srad_wmm,
    avg(d2.tmmn) AS temp_min_k,
    avg(d2.tmmx) AS temp_max_k,
    avg(d2.vpd) AS mean_vapor_pressure_deficit_kpa,
    avg(d2.vs) AS wind_speed_10m_m_s
from 
	wildfire.fod as fod
right join
	wildfire.fpi as fpi on fod.fire_date = fpi.fpi_date and fire_lat_rnd = fpi_lat_rnd and fire_lon_rnd = fpi_lon_rnd
JOIN 
        wildfire.geo_combined d2 ON fpi.fpi_date = d2.date AND fpi.fpi_lat_rnd = round(d2.latitude, 1) AND fpi.fpi_lon_rnd = round(d2.longitude, 1)
LEFT JOIN 
	staging.elevation e ON fpi.fpi_lat_rnd = e.latitude::numeric AND fpi.fpi_lon_rnd = e.longitude::numeric
group by
	1, 2, 3
;

CREATE INDEX comb_fire_lat_idx ON wildfire.combined (lat_rnd);
CREATE INDEX comb_fire_lon_idx ON wildfire.combined (lon_rnd);
CREATE INDEX comb_fire_year_idx ON wildfire.combined (fpi_date);
CREATE INDEX comb_fire_date_idx ON wildfire.combined (fire_occurrence);

GRANT SELECT ON TABLE wildfire.combined TO students;

refresh materialized view wildfire.combined;
