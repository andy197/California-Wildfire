drop materialized view if exists results.all_models;
create materialized view results.all_models as 
select
	el.lat_rnd,
	el.lon_rnd,
	el.fpi_date,
	el.no_fire as el_no_fire,
	el.fire as el_fire,
	nn.prob_no_fire::float8 as nn_no_fire,
	nn.prob_fire::float8 as nn_fire,
	fl.no_fire as lr_no_fire,
	fl.fire as lr_fire,
	r.rf_nofire_prob as rf_no_fire,
	r.rf_fire_prob as rf_fire
from
	results.ensemble_le el
join
	results."2015_neural_net" nn on nn.lat_rnd::numeric = el.lat_rnd and nn.lon_rnd::numeric = el.lon_rnd and nn.fpi_date::date = el.fpi_date
join 
	results.fpi_lr fl on fl.lat_rnd = el.lat_rnd and fl.lon_rnd = el.lon_rnd and fl.fpi_date = el.fpi_date
join 
	results.randomforest r on r.lat_rnd::numeric = el.lat_rnd and r.lon_rnd::numeric = el.lon_rnd and r.fpi_date = el.fpi_date
;

GRANT SELECT ON TABLE results.all_models TO demo;
GRANT SELECT ON TABLE results.all_models TO students;
