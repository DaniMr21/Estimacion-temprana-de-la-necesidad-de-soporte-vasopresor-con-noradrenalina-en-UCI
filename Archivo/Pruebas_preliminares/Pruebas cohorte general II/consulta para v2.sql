drop table if exists public.cohorte_norad_v2;

create table public.cohorte_norad_v2 as
select 
    c.subject_id,
    c.hadm_id,
    c.stay_id,
    c.intime,
    c.outtime,
    c.anchor_age,
    c.contador_estancia_uci,

    n.inicio_norad,

    extract(epoch from (n.inicio_norad - c.intime)) / 3600.0 as horas_hasta_norad

from public.cohorte_base_v2 c

left join public.noradrenalina_v2 n
    on c.stay_id = n.stay_id;

select *
from public.cohorte_norad_v2 where horas_hasta_norad <0
limit 20;

drop table if exists public.cohorte_norad_v2_limpio;

create table public.cohorte_norad_v2_limpio as
select * from public.cohorte_norad_v2
where horas_hasta_norad is null or horas_hasta_norad >= 0;

select *
from public.cohorte_norad_v2_limpio
where horas_hasta_norad < 0
limit 10;

drop table if exists public.dataset_modelo_v2;

create table public.dataset_modelo_v2 as
select *,
    case
        when horas_hasta_norad between 6 and 24 then 1
        else 0
    end as etiqueta_norad_6_24
from public.cohorte_norad_v2_limpio;

select etiqueta_norad_6_24, count(*) as n
from public.dataset_modelo_v2
group by etiqueta_norad_6_24;

drop table if exists public.dataset_modelo_v2_6h;

create table public.dataset_modelo_v2_6h as
select *
from public.dataset_modelo_v2
where 
    horas_hasta_norad is null
    or horas_hasta_norad >= 6;

select 
    etiqueta_norad_6_24,
    count(*) as n
from public.dataset_modelo_v2_6h group by etiqueta_norad_6_24;

select 
    count(*) as total_filas,
    count(distinct subject_id) as pacientes_unicos,
    count(distinct stay_id) as estancias_unicas,
    avg(etiqueta_norad_6_24::float) as prevalencia
from public.dataset_modelo_v2_6h;

select 
    contador_estancia_uci,
    count(*) as n
from public.dataset_modelo_v2_6h
group by contador_estancia_uci
order by contador_estancia_uci
limit 10;

select 
    itemid,
    label,
    fluid,
    category
from mimiciv_hosp.d_labitems
where itemid = 51301;

select 
    subject_id,
    hadm_id,
    charttime,
    storetime,
    itemid,
    valuenum,
    valueuom,
    flag
from mimiciv_hosp.labevents
where itemid = 51301
limit 20;

drop table if exists public.leucocitos_v2;

create table public.leucocitos_v2 as
select 
    c.stay_id,

    min(l.valuenum) as leucocitos_min,
    max(l.valuenum) as leucocitos_max,
    avg(l.valuenum) as leucocitos_media

from public.dataset_modelo_v2_6h c

left join mimiciv_hosp.labevents l
    on c.hadm_id = l.hadm_id
    and l.itemid = 51301
    and l.valuenum is not null
    and l.charttime >= c.intime
    and l.charttime <= c.intime + interval '6 hour'

group by c.stay_id;

select 
    count(*) as total_estancias,
    count(leucocitos_media) as con_dato,
    count(*) - count(leucocitos_media) as missing,
    100.0 * (count(*) - count(leucocitos_media)) / count(*) as pct_missing
from public.leucocitos_v2;

select 
    itemid,
    label,
    fluid,
    category
from mimiciv_hosp.d_labitems
where itemid = 50820;

select 
    subject_id,
    hadm_id,
    charttime,
    storetime,
    itemid,
    valuenum,
    valueuom,
    flag
from mimiciv_hosp.labevents
where itemid = 50820 

drop table if exists public.ph_v2_6h;

create table public.ph_v2_6h as
select 
    c.stay_id,
    min(l.valuenum) as ph_min,
    max(l.valuenum) as ph_max,
    avg(l.valuenum) as ph_media
from public.dataset_modelo_v2_6h c
left join mimiciv_hosp.labevents l
    on c.hadm_id = l.hadm_id
    and l.itemid = 50820
    and l.valuenum is not null
    and l.charttime >= c.intime
    and l.charttime <= c.intime + interval '6 hour'
group by c.stay_id;

select 
    count(*) as total_estancias,
    count(ph_media) as con_dato,
    count(*) - count(ph_media) as missing,
    100.0 * (count(*) - count(ph_media)) / count(*) as pct_missing
from public.ph_v2_6h;

select 
    itemid,
    label,
    fluid,
    category
from mimiciv_hosp.d_labitems
where itemid = 51256;

select 
    subject_id,
    hadm_id,
    charttime,
    storetime,
    itemid,
    valuenum,
    valueuom,
    flag
from mimiciv_hosp.labevents
where itemid = 51256
limit 20;

drop table if exists public.neutro_v2_6h;

create table public.neutro_v2_6h as
select 
    c.stay_id,
    min(l.valuenum) as neutro_min,
    max(l.valuenum) as neutro_max,
    avg(l.valuenum) as neutro_media
from public.dataset_modelo_v2_6h c
left join mimiciv_hosp.labevents l
    on c.hadm_id = l.hadm_id
    and l.itemid = 51256
    and l.valuenum is not null
    and l.charttime >= c.intime
    and l.charttime <= c.intime + interval '6 hour'
group by c.stay_id;

select 
    count(*) as total_estancias,
    count(neutro_media) as con_dato,
    count(*) - count(neutro_media) as missing,
    100.0 * (count(*) - count(neutro_media)) / count(*) as pct_missing
from public.neutro_v2_6h;

select 
    itemid,
    label,
    fluid,
    category
from mimiciv_hosp.d_labitems
where itemid = 50889;

select 
    subject_id,
    hadm_id,
    charttime,
    storetime,
    itemid,
    valuenum,
    valueuom,
    flag
from mimiciv_hosp.labevents
where itemid = 50889
limit 20;

drop table if exists public.pcr_v2_6h;

create table public.pcr_v2_6h as
select 
    c.stay_id,
    min(l.valuenum) as pcr_min,
    max(l.valuenum) as pcr_max,
    avg(l.valuenum) as pcr_media
from public.dataset_modelo_v2_6h c
left join mimiciv_hosp.labevents l
    on c.hadm_id = l.hadm_id
    and l.itemid = 50889
    and l.valuenum is not null
    and l.charttime >= c.intime
    and l.charttime <= c.intime + interval '6 hour'
group by c.stay_id;

select 
    count(*) as total_estancias,
    count(pcr_media) as con_dato,
    count(*) - count(pcr_media) as missing,
    100.0 * (count(*) - count(pcr_media)) / count(*) as pct_missing
from public.pcr_v2_6h;

select 
    itemid,
    label,
    fluid,
    category
from mimiciv_hosp.d_labitems
where itemid = 50821;

select 
    subject_id,
    hadm_id,
    charttime,
    storetime,
    itemid,
    valuenum,
    valueuom,
    flag
from mimiciv_hosp.labevents
where itemid = 50821
limit 20;

drop table if exists public.pao2_v2_6h;

create table public.pao2_v2_6h as
select 
    c.stay_id,
    min(l.valuenum) as pao2_min,
    max(l.valuenum) as pao2_max,
    avg(l.valuenum) as pao2_media
from public.dataset_modelo_v2_6h c
left join mimiciv_hosp.labevents l
    on c.hadm_id = l.hadm_id
    and l.itemid = 50821
    and l.valuenum is not null
    and l.charttime >= c.intime
    and l.charttime <= c.intime + interval '6 hour'
group by c.stay_id;

select 
    count(*) as total_estancias,
    count(pao2_media) as con_dato,
    count(*) - count(pao2_media) as missing,
    100.0 * (count(*) - count(pao2_media)) / count(*) as pct_missing
from public.pao2_v2_6h;

select 
    itemid,
    label,
    fluid,
    category
from mimiciv_hosp.d_labitems
where itemid = 50878;

select 
    subject_id,
    hadm_id,
    charttime,
    storetime,
    itemid,
    valuenum,
    valueuom,
    flag
from mimiciv_hosp.labevents
where itemid = 50878
limit 20;

drop table if exists public.got_v2_6h;

create table public.got_v2_6h as
select 
    c.stay_id,
    min(l.valuenum) as got_min,
    max(l.valuenum) as got_max,
    avg(l.valuenum) as got_media
from public.dataset_modelo_v2_6h c
left join mimiciv_hosp.labevents l
    on c.hadm_id = l.hadm_id
    and l.itemid = 50878
    and l.valuenum is not null
    and l.charttime >= c.intime
    and l.charttime <= c.intime + interval '6 hour'
group by c.stay_id;

select 
    count(*) as total_estancias,
    count(got_media) as con_dato,
    count(*) - count(got_media) as missing,
    100.0 * (count(*) - count(got_media)) / count(*) as pct_missing
from public.got_v2_6h;

select 
    itemid,
    label,
    fluid,
    category
from mimiciv_hosp.d_labitems
where itemid = 50861;

select 
    subject_id,
    hadm_id,
    charttime,
    storetime,
    itemid,
    valuenum,
    valueuom,
    flag
from mimiciv_hosp.labevents
where itemid = 50861
limit 20;

drop table if exists public.gpt_v2_6h;

create table public.gpt_v2_6h as
select 
    c.stay_id,
    min(l.valuenum) as gpt_min,
    max(l.valuenum) as gpt_max,
    avg(l.valuenum) as gpt_media
from public.dataset_modelo_v2_6h c
left join mimiciv_hosp.labevents l
    on c.hadm_id = l.hadm_id
    and l.itemid = 50861
    and l.valuenum is not null
    and l.charttime >= c.intime
    and l.charttime <= c.intime + interval '6 hour'
group by c.stay_id;

select 
    count(*) as total_estancias,
    count(gpt_media) as con_dato,
    count(*) - count(gpt_media) as missing,
    100.0 * (count(*) - count(gpt_media)) / count(*) as pct_missing
from public.gpt_v2_6h;

select 
    itemid,
    label,
    fluid,
    category
from mimiciv_hosp.d_labitems
where itemid = 50954;

select 
    subject_id,
    hadm_id,
    charttime,
    storetime,
    itemid,
    valuenum,
    valueuom,
    flag
from mimiciv_hosp.labevents
where itemid = 50954
limit 20;

drop table if exists public.ldh_v2_6h;

create table public.ldh_v2_6h as
select 
    c.stay_id,
    min(l.valuenum) as ldh_min,
    max(l.valuenum) as ldh_max,
    avg(l.valuenum) as ldh_media
from public.dataset_modelo_v2_6h c
left join mimiciv_hosp.labevents l
    on c.hadm_id = l.hadm_id
    and l.itemid = 50954
    and l.valuenum is not null
    and l.charttime >= c.intime
    and l.charttime <= c.intime + interval '6 hour'
group by c.stay_id;

select 
    count(*) as total_estancias,
    count(ldh_media) as con_dato,
    count(*) - count(ldh_media) as missing,
    100.0 * (count(*) - count(ldh_media)) / count(*) as pct_missing
from public.ldh_v2_6h;

select 
    itemid,
    label,
    fluid,
    category
from mimiciv_hosp.d_labitems
where itemid = 51274;

select 
    subject_id,
    hadm_id,
    charttime,
    storetime,
    itemid,
    valuenum,
    valueuom,
    flag
from mimiciv_hosp.labevents
where itemid = 51274
limit 20;

--Aquí viene cuando la matan---------------------------------

drop table if exists public.datos_completos_6h;

create table public.datos_completos_6h as
select 
    c.stay_id,

    -- Lactato
    min(lactato.valuenum) as lactato_min,
    max(lactato.valuenum) as lactato_max,
    avg(lactato.valuenum) as lactato_media,

    -- Creatinina
    min(creatinina.valuenum) as creatinina_min,
    max(creatinina.valuenum) as creatinina_max,
    avg(creatinina.valuenum) as creatinina_media,

    -- Plaquetas
    min(plaquetas.valuenum) as plaquetas_min,
    max(plaquetas.valuenum) as plaquetas_max,
    avg(plaquetas.valuenum) as plaquetas_media,

    -- Bilirrubina
    min(bilirrubina.valuenum) as bilirrubina_min,
    max(bilirrubina.valuenum) as bilirrubina_max,
    avg(bilirrubina.valuenum) as bilirrubina_media

from public.dataset_modelo_v2_6h c

left join mimiciv_hosp.labevents lactato
    on c.hadm_id = lactato.hadm_id
    and lactato.itemid = 50813
    and lactato.valuenum is not null
    and lactato.charttime >= c.intime
    and lactato.charttime <= c.intime + interval '6 hour'

left join mimiciv_hosp.labevents creatinina
    on c.hadm_id = creatinina.hadm_id
    and creatinina.itemid = 50912
    and creatinina.valuenum is not null
    and creatinina.charttime >= c.intime
    and creatinina.charttime <= c.intime + interval '6 hour'

left join mimiciv_hosp.labevents plaquetas
    on c.hadm_id = plaquetas.hadm_id
    and plaquetas.itemid = 51265
    and plaquetas.valuenum is not null
    and plaquetas.charttime >= c.intime
    and plaquetas.charttime <= c.intime + interval '6 hour'

left join mimiciv_hosp.labevents bilirrubina
    on c.hadm_id = bilirrubina.hadm_id
    and bilirrubina.itemid = 50885
    and bilirrubina.valuenum is not null
    and bilirrubina.charttime >= c.intime
    and bilirrubina.charttime <= c.intime + interval '6 hour'

group by c.stay_id;

-- Unir todas las tablas creadas y obtener los datos finales
-- Unir todas las tablas calculadas con la tabla de constantes_6h
drop table if exists public.datos_finales_6h;

create table public.datos_finales_6h as
select 
    d.*,

    -- Vitales
    v.media_card,
    v.min_card,
    v.max_card,
    v.media_resp,
    v.temp_media,
    v.spo2_media,
    v.map_media,
    v.map_min,

    -- Analíticas individuales
    tp.tp_media,
    ldh.ldh_media,
    gpt.gpt_media,
    got.got_media,
    pao2.pao2_media,
    pcr.pcr_media,
    neutro.neutro_media,
    ph.ph_media,
    leu.leucocitos_media

from public.datos_completos_6h d

left join public.vitales_6h v        
    on d.stay_id = v.stay_id
left join public.tp_v2_6h tp
    on d.stay_id = tp.stay_id
left join public.ldh_v2_6h ldh
    on d.stay_id = ldh.stay_id
left join public.gpt_v2_6h gpt
    on d.stay_id = gpt.stay_id
left join public.got_v2_6h got
    on d.stay_id = got.stay_id
left join public.pao2_v2_6h pao2
    on d.stay_id = pao2.stay_id
left join public.pcr_v2_6h pcr
    on d.stay_id = pcr.stay_id
left join public.neutro_v2_6h neutro
    on d.stay_id = neutro.stay_id
left join public.ph_v2_6h ph
    on d.stay_id = ph.stay_id
left join public.leucocitos_v2 leu
    on d.stay_id = leu.stay_id;


--EXPLORAMOS 

select *
from public.datos_finales_6h LIMIT 5;

select
    count(*) as total_filas,

    -- Lactato
    round(100.0 * count(lactato_min) / count(*), 1) as pct_lactato_min,
    round(100.0 * count(lactato_max) / count(*), 1) as pct_lactato_max,
    round(100.0 * count(lactato_media) / count(*), 1) as pct_lactato_media,

    -- Creatinina
    round(100.0 * count(creatinina_min) / count(*), 1) as pct_creatinina_min,
    round(100.0 * count(creatinina_max) / count(*), 1) as pct_creatinina_max,
    round(100.0 * count(creatinina_media) / count(*), 1) as pct_creatinina_media,

    -- Plaquetas
    round(100.0 * count(plaquetas_min) / count(*), 1) as pct_plaquetas_min,
    round(100.0 * count(plaquetas_max) / count(*), 1) as pct_plaquetas_max,
    round(100.0 * count(plaquetas_media) / count(*), 1) as pct_plaquetas_media,

    -- Bilirrubina
    round(100.0 * count(bilirrubina_min) / count(*), 1) as pct_bilirrubina_min,
    round(100.0 * count(bilirrubina_max) / count(*), 1) as pct_bilirrubina_max,
    round(100.0 * count(bilirrubina_media) / count(*), 1) as pct_bilirrubina_media,

    -- Heart rate
    round(100.0 * count(media_card) / count(*), 1) as pct_hr_media,
    round(100.0 * count(min_card) / count(*), 1) as pct_hr_min,
    round(100.0 * count(max_card) / count(*), 1) as pct_hr_max,

    -- Resp rate
    round(100.0 * count(media_resp) / count(*), 1) as pct_resp_media,

    -- Temperatura
    round(100.0 * count(temp_media) / count(*), 1) as pct_temp_media,

    -- SpO2
    round(100.0 * count(spo2_media) / count(*), 1) as pct_spo2_media,

    -- MAP
    round(100.0 * count(map_media) / count(*), 1) as pct_map_media,
    round(100.0 * count(map_min) / count(*), 1) as pct_map_min,

    -- TP
    round(100.0 * count(tp_media) / count(*), 1) as pct_tp_media,

    -- LDH
    round(100.0 * count(ldh_media) / count(*), 1) as pct_ldh_media,

    -- GPT
    round(100.0 * count(gpt_media) / count(*), 1) as pct_gpt_media,

    -- GOT
    round(100.0 * count(got_media) / count(*), 1) as pct_got_media,

    -- PaO2
    round(100.0 * count(pao2_media) / count(*), 1) as pct_pao2_media,

    -- PCR
    round(100.0 * count(pcr_media) / count(*), 1) as pct_pcr_media,

    -- Neutrófilos
    round(100.0 * count(neutro_media) / count(*), 1) as pct_neutro_media,

    -- pH
    round(100.0 * count(ph_media) / count(*), 1) as pct_ph_media,

    -- Leucocitos
    round(100.0 * count(leucocitos_media) / count(*), 1) as pct_leucocitos_media



select count(*) 
from public.datos_finales_6h
where lactato_media is not null
and creatinina_media is not null
and plaquetas_media is not null
and bilirrubina_media is not null
and media_card is not null
and min_card is not null
and max_card is not null
and media_resp is not null
and temp_media is not null
and spo2_media is not null
and map_media is not null
and map_min is not null
and tp_media is not null
and ldh_media is not null
and gpt_media is not null
and got_media is not null
and pao2_media is not null
and neutro_media is not null
and ph_media is not null
and leucocitos_media is not null;

--unir con los pacientes de la cohorte:

drop table if exists public.dataset_final_v2;

create table public.dataset_final_v2 as
select
    -- Identificadores y variables de cohorte
    c.subject_id,
    c.hadm_id,
    c.stay_id,
    c.anchor_age,
    c.contador_estancia_uci,
    c.horas_hasta_norad,
    c.etiqueta_norad_6_24,

    -- Variables clínicas
    d.lactato_min,
    d.lactato_max,
    d.lactato_media,
    d.creatinina_min,
    d.creatinina_max,
    d.creatinina_media,
    d.plaquetas_min,
    d.plaquetas_max,
    d.plaquetas_media,
    d.bilirrubina_min,
    d.bilirrubina_max,
    d.bilirrubina_media,
    d.media_card,
    d.min_card,
    d.max_card,
    d.media_resp,
    d.temp_media,
    d.spo2_media,
    d.map_media,
    d.map_min,
    d.tp_media,
    d.gpt_media,
    d.got_media,
    d.pao2_media,
    d.neutro_media,
    d.ph_media,
    d.leucocitos_media

from public.dataset_modelo_v2_6h c
left join public.datos_finales_6h d
    on c.stay_id = d.stay_id;

-- Verificación
select 
    count(*) as total_filas,
    count(distinct subject_id) as pacientes_unicos,
    count(distinct stay_id) as estancias_unicas,
    sum(etiqueta_norad_6_24) as positivos
from public.dataset_final_v2.

select * from public.dataset_final_v2;

-------------------------------------------------tabla sin missings

drop table if exists public.dataset_final_v2_sinmissing;

create table public.dataset_final_v2_sinmissing as
select * from public.dataset_final_v2
where lactato_media is not null
and creatinina_media is not null
and plaquetas_media is not null
and bilirrubina_media is not null
and media_card is not null
and min_card is not null
and max_card is not null
and media_resp is not null
and temp_media is not null
and spo2_media is not null
and map_media is not null
and map_min is not null
and tp_media is not null
and gpt_media is not null
and got_media is not null
and pao2_media is not null
and neutro_media is not null
and ph_media is not null
and leucocitos_media is not null;

-- Verificación
select count(*) as total_filas, sum(etiqueta_norad_6_24) as positivos, round(avg(etiqueta_norad_6_24) * 100, 2) as prevalencia_pct
from public.dataset_final_v2_sinmissing;

select * from public.dataset_final_v2_sinmissing;