-- =============================================================================
-- DATASET VALIDACIÓN EXTERNA eICU — ventana 12-48h
-- Modelo: XGBoost Largo_12_48
-- Variables del modelo: temp_min, pf_min, spo2_min, bicarbonato_min,
--                       map_min, glucemia_min, sofa_max
-- =============================================================================


-- 1. COHORTE BASE v4l: adultos ≥18, estancia UCI >48h
drop table if exists cohorte_base_eicu_v4l;
create table cohorte_base_eicu_v4l as
select p.patientunitstayid, p.uniquepid, p.patienthealthsystemstayid,
    case
        when p.age = '> 89' then 90
        when p.age ~ '^\d+$' then cast(p.age as integer)
        else null
    end as anchor_age,
    p.gender,
    p.unitvisitnumber as contador_estancia_uci,
    p.unitdischargeoffset,
    coalesce(p.admissionweight, p.dischargeweight) as peso_kg
from "patient" p
where (p.age = '> 89' or (p.age ~ '^\d+$' and cast(p.age as integer) >= 18))
  and p.unitdischargeoffset > 2880;


-- 2. COHORTE + NORAD v4l (reutiliza noradrenalina_eicu del script v4)
drop table if exists cohorte_norad_eicu_v4l;
create table cohorte_norad_eicu_v4l as
select c.*, n.inicio_norad_min, n.inicio_norad_min / 60.0 as horas_hasta_norad
from cohorte_base_eicu_v4l c
left join noradrenalina_eicu n on c.patientunitstayid = n.patientunitstayid
where n.inicio_norad_min is null or n.inicio_norad_min >= 0;


-- 3. DATASET MODELO v4l
drop table if exists dataset_modelo_eicu_v4l;
create table dataset_modelo_eicu_v4l as
select *,
    case
        when horas_hasta_norad between 12 and 48 then 1
        else 0
    end as etiqueta_norad_12_48
from cohorte_norad_eicu_v4l
where horas_hasta_norad is null or horas_hasta_norad >= 12;


-- 4. LABORATORIO (ventana 0-12h)
-- bicarbonato_min, glucemia_min: variables del modelo
-- pao2_min: para calcular pf_min
-- bilirrubina, plaquetas, creatinina: auxiliares SOFA únicamente
drop table if exists variables_lab_eicu_v4l;
create table variables_lab_eicu_v4l as
select c.patientunitstayid,
    min(case when l.labname = 'bicarbonate'      then l.labresult end) as bicarbonato_min,
    min(case when l.labname = 'glucose'          then l.labresult end) as glucemia_min,
    min(case when l.labname = 'paO2'             then l.labresult end) as pao2_min,
    -- Auxiliares para SOFA
    max(case when l.labname = 'total bilirubin'  then l.labresult end) as bilirrubina_max,
    min(case when l.labname = 'platelets x 1000' then l.labresult end) as plaquetas_min,
    max(case when l.labname = 'creatinine'       then l.labresult end) as creatinina_max
from dataset_modelo_eicu_v4l c
left join "lab" l on c.patientunitstayid = l.patientunitstayid
    and l.labresult is not null
    and l.labresultoffset between 0 and 720
    and l.labname in (
        'bicarbonate', 'glucose', 'paO2',
        'total bilirubin', 'platelets x 1000', 'creatinine'
    )
group by c.patientunitstayid;


-- 5. VITALES (ventana 0-12h) — temp_min, spo2_min, map_min
drop table if exists variables_vitales_eicu_v4l;
create table variables_vitales_eicu_v4l as
with periodic as (
    select c.patientunitstayid,
        min(vp.sao2)         as spo2_min,
        min(vp.systemicmean) as map_art_min,
        min(case when vp.temperature > 50 then (vp.temperature - 32) * 5.0/9.0
                 else vp.temperature end) as temp_min_periodic
    from dataset_modelo_eicu_v4l c
    left join "vitalPeriodic" vp on c.patientunitstayid = vp.patientunitstayid
        and vp.observationoffset between 0 and 720
    group by c.patientunitstayid
),
aperiodic as (
    select c.patientunitstayid,
        min(va.noninvasivemean) as map_nibp_min
    from dataset_modelo_eicu_v4l c
    left join "vitalAperiodic" va on c.patientunitstayid = va.patientunitstayid
        and va.observationoffset between 0 and 720
    group by c.patientunitstayid
),
temp_nurse as (
    select c.patientunitstayid,
        min(case
            when nc.nursingchartcelltypevalname = 'Temperature (F)'
                 and nc.nursingchartvalue ~ '^\d+(\.\d+)?$'
            then (cast(nc.nursingchartvalue as numeric) - 32) * 5.0/9.0
            when nc.nursingchartcelltypevalname = 'Temperature (C)'
                 and nc.nursingchartvalue ~ '^\d+(\.\d+)?$'
            then cast(nc.nursingchartvalue as numeric)
        end) as temp_min_nurse
    from dataset_modelo_eicu_v4l c
    left join "nurseCharting" nc on c.patientunitstayid = nc.patientunitstayid
        and nc.nursingchartoffset between 0 and 720
        and nc.nursingchartcelltypevalname in ('Temperature (C)', 'Temperature (F)')
    group by c.patientunitstayid
)
select p.patientunitstayid,
    p.spo2_min,
    coalesce(p.temp_min_periodic, t.temp_min_nurse) as temp_min,
    coalesce(p.map_art_min, a.map_nibp_min)         as map_min
from periodic p
left join aperiodic  a on p.patientunitstayid = a.patientunitstayid
left join temp_nurse t on p.patientunitstayid = t.patientunitstayid;


-- 6. FiO2 NORMALIZADO (ventana 0-12h) — para pf_min y SOFA respiratorio
drop table if exists variables_fio2_eicu_v4l;
create table variables_fio2_eicu_v4l as
with raw as (
    select c.patientunitstayid,
        case when rc.respchartvalue ~ '^\d+(\.\d+)?$'
             then cast(rc.respchartvalue as numeric) end as fio2_raw
    from dataset_modelo_eicu_v4l c
    left join "respiratoryCharting" rc on c.patientunitstayid = rc.patientunitstayid
        and rc.respchartoffset between 0 and 720
        and rc.respchartvaluelabel in ('FiO2', 'FIO2 (%)', 'Set Fraction of Inspired Oxygen (FIO2)')
),
normalizado as (
    select patientunitstayid,
        case
            when fio2_raw between 0.21 and 1.0 then fio2_raw * 100
            when fio2_raw between 21   and 100  then fio2_raw
            else null
        end as fio2_pct
    from raw
)
select patientunitstayid, max(fio2_pct) as fio2_max
from normalizado where fio2_pct is not null
group by patientunitstayid;


-- 7. GCS (ventana 0-12h) — solo para SOFA neurológico
drop table if exists variables_gcs_eicu_v4l;
create table variables_gcs_eicu_v4l as
select c.patientunitstayid,
    min(case when nc.nursingchartvalue ~ '^\d+$'
             then cast(nc.nursingchartvalue as numeric) end) as gcs_min
from dataset_modelo_eicu_v4l c
left join "nurseCharting" nc on c.patientunitstayid = nc.patientunitstayid
    and nc.nursingchartoffset between 0 and 720
    and nc.nursingchartcelltypevalname = 'GCS Total'
group by c.patientunitstayid;


-- 8. SOFA (ventana 0-12h) — variable del modelo
drop table if exists variables_sofa_eicu_v4l;
create table variables_sofa_eicu_v4l as
with vasopresores as (
    select c.patientunitstayid,
        max(case when lower(id.drugname) like '%norepinephrine%'
                   or lower(id.drugname) like '%levophed%'          then 1 else 0 end) as con_norad,
        max(case when lower(id.drugname) like '%epinephrine%'
                  and lower(id.drugname) not like '%norepinephrine%' then 1 else 0 end) as con_epi,
        max(case when lower(id.drugname) like '%dopamine%'          then 1 else 0 end) as con_dopa,
        max(case when lower(id.drugname) like '%dobutamine%'        then 1 else 0 end) as con_dobu
    from dataset_modelo_eicu_v4l c
    left join "infusionDrug" id on c.patientunitstayid = id.patientunitstayid
        and id.infusionoffset between 0 and 720
        and lower(id.drugname) not like '%volume%'
    group by c.patientunitstayid
),
componentes as (
    select c.patientunitstayid,
        case when l.pao2_min is not null and f.fio2_max is not null and f.fio2_max > 0
             then l.pao2_min / (f.fio2_max / 100.0) else null end   as pf_min,
        case
            when v.con_norad = 1 or v.con_epi  = 1 then 3
            when v.con_dopa  = 1 or v.con_dobu = 1 then 2
            when vt.map_min is not null and vt.map_min < 70 then 1
            when vt.map_min is not null then 0
            else null
        end                                                          as cardiovascular,
        case
            when g.gcs_min >= 13 and g.gcs_min <= 14 then 1
            when g.gcs_min >= 10 and g.gcs_min <= 12 then 2
            when g.gcs_min >= 6  and g.gcs_min <= 9  then 3
            when g.gcs_min < 6                        then 4
            when g.gcs_min is null                    then null else 0
        end                                                          as cns,
        case
            when l.bilirrubina_max >= 12.0 then 4
            when l.bilirrubina_max >=  6.0 then 3
            when l.bilirrubina_max >=  2.0 then 2
            when l.bilirrubina_max >=  1.2 then 1
            when l.bilirrubina_max is null  then null else 0
        end                                                          as liver,
        case
            when l.plaquetas_min <  20 then 4
            when l.plaquetas_min <  50 then 3
            when l.plaquetas_min < 100 then 2
            when l.plaquetas_min < 150 then 1
            when l.plaquetas_min is null then null else 0
        end                                                          as coagulation,
        case
            when l.creatinina_max >= 5.0 then 4
            when l.creatinina_max >= 3.5 then 3
            when l.creatinina_max >= 2.0 then 2
            when l.creatinina_max >= 1.2 then 1
            when l.creatinina_max is null then null else 0
        end                                                          as renal
    from dataset_modelo_eicu_v4l c
    left join variables_lab_eicu_v4l      l  on c.patientunitstayid = l.patientunitstayid
    left join variables_fio2_eicu_v4l     f  on c.patientunitstayid = f.patientunitstayid
    left join variables_gcs_eicu_v4l      g  on c.patientunitstayid = g.patientunitstayid
    left join variables_vitales_eicu_v4l  vt on c.patientunitstayid = vt.patientunitstayid
    left join vasopresores                v  on c.patientunitstayid = v.patientunitstayid
)
select patientunitstayid,
    coalesce(case when pf_min < 100 then 4 when pf_min < 200 then 3
                  when pf_min < 300 then 2 when pf_min < 400 then 1
                  else 0 end, 0)
    + coalesce(cardiovascular, 0)
    + coalesce(cns, 0)
    + coalesce(liver, 0)
    + coalesce(coagulation, 0)
    + coalesce(renal, 0) as sofa_max
from componentes;


-- 9. TABLA FINAL — identificadores + etiqueta + 7 variables del modelo
drop table if exists dataset_final_eicu_v4l;
create table dataset_final_eicu_v4l as
select
    c.uniquepid              as subject_id,
    c.patientunitstayid      as stay_id,
    c.anchor_age,
    c.gender,
    c.peso_kg,
    c.contador_estancia_uci,
    c.horas_hasta_norad,
    c.etiqueta_norad_12_48,
    -- 7 variables del modelo XGB Largo
    vt.temp_min,
    case
        when l.pao2_min is not null and f.fio2_max is not null and f.fio2_max > 0
        then l.pao2_min / (f.fio2_max / 100.0)
        else null
    end                      as pf_min,
    vt.spo2_min,
    l.bicarbonato_min,
    vt.map_min,
    l.glucemia_min,
    so.sofa_max
from dataset_modelo_eicu_v4l      c
left join variables_lab_eicu_v4l      l  on c.patientunitstayid = l.patientunitstayid
left join variables_vitales_eicu_v4l  vt on c.patientunitstayid = vt.patientunitstayid
left join variables_fio2_eicu_v4l     f  on c.patientunitstayid = f.patientunitstayid
left join variables_sofa_eicu_v4l     so on c.patientunitstayid = so.patientunitstayid;


-- 10. TABLA FINAL LIMPIA — solo filas completas en las 7 variables del modelo
drop table if exists dataset_final_eicu_v4l_clean;
create table dataset_final_eicu_v4l_clean as
select * from dataset_final_eicu_v4l
where temp_min        is not null
  and pf_min          is not null
  and spo2_min        is not null
  and bicarbonato_min is not null
  and map_min         is not null
  and glucemia_min    is not null
  and sofa_max        is not null;


-- VERIFICACIÓN
select
    count(*)                                                as total_filas,
    count(distinct subject_id)                              as pacientes_unicos,
    sum(etiqueta_norad_12_48)                               as positivos,
    round(100.0 * sum(etiqueta_norad_12_48) / count(*), 2) as prevalencia_pct
from dataset_final_eicu_v4l_clean;


-- DESCARGAR
select * from dataset_final_eicu_v4l_clean;
