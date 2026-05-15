-- =============================================================================
-- DATASET VALIDACIÓN EXTERNA eICU — ventana 3-12h
-- Modelo: CatBoost Corto_3_12
-- Variables del modelo: map_min, pf_min, sofa_max, tp_max
-- =============================================================================


-- 1. COHORTE BASE
drop table if exists dataset_modelo_eicu_v4p;
create table dataset_modelo_eicu_v4p as
select *,
    case
        when horas_hasta_norad between 3 and 12 then 1
        else 0
    end as etiqueta_norad_3_12
from cohorte_norad_eicu where horas_hasta_norad is null or horas_hasta_norad >= 3;


-- 2. LABORATORIO (ventana 0-3h)
-- tp_max: variable del modelo
-- pao2, bilirrubina, plaquetas, creatinina: auxiliares para SOFA interno
drop table if exists variables_lab_eicu_v4p;
create table variables_lab_eicu_v4p as
select c.patientunitstayid,
    max(case when l.labname in ('PT', 'PT - protime', 'protime')
             then l.labresult end)                           as tp_max,
    min(case when l.labname = 'paO2'
             then l.labresult end)                           as pao2_min,
    max(case when l.labname = 'total bilirubin'
             then l.labresult end)                           as bilirrubina_max,
    min(case when l.labname = 'platelets x 1000'
             then l.labresult end)                           as plaquetas_min,
    max(case when l.labname = 'creatinine'
             then l.labresult end)                           as creatinina_max
from dataset_modelo_eicu_v4p c
left join "lab" l on c.patientunitstayid = l.patientunitstayid
    and l.labresult is not null
    and l.labresultoffset between 0 and 180
    and l.labname in (
        'PT', 'PT - protime', 'protime',
        'paO2',
        'total bilirubin', 'platelets x 1000', 'creatinine'
    )
group by c.patientunitstayid;


-- 3. VITALES (ventana 0-3h) — solo map_min
drop table if exists variables_vitales_eicu_v4p;
create table variables_vitales_eicu_v4p as
with periodic as (
    select c.patientunitstayid,
        min(vp.systemicmean) as map_art_min
    from dataset_modelo_eicu_v4p c
    left join "vitalPeriodic" vp on c.patientunitstayid = vp.patientunitstayid
        and vp.observationoffset between 0 and 180
    group by c.patientunitstayid
),
aperiodic as (
    select c.patientunitstayid,
        min(va.noninvasivemean) as map_nibp_min
    from dataset_modelo_eicu_v4p c
    left join "vitalAperiodic" va on c.patientunitstayid = va.patientunitstayid
        and va.observationoffset between 0 and 180
    group by c.patientunitstayid
)
select p.patientunitstayid,
    coalesce(p.map_art_min, a.map_nibp_min) as map_min
from periodic p
left join aperiodic a on p.patientunitstayid = a.patientunitstayid;


-- 4. FiO2 NORMALIZADO (ventana 0-3h) — para pf_min y SOFA respiratorio
drop table if exists variables_fio2_eicu_v4p;
create table variables_fio2_eicu_v4p as
with raw as (
    select c.patientunitstayid,
        case when rc.respchartvalue ~ '^\d+(\.\d+)?$'
             then cast(rc.respchartvalue as numeric) end as fio2_raw
    from dataset_modelo_eicu_v4p c
    left join "respiratoryCharting" rc on c.patientunitstayid = rc.patientunitstayid
        and rc.respchartoffset between 0 and 180
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


-- 5. GCS (ventana 0-3h) — solo para SOFA neurológico
drop table if exists variables_gcs_eicu_v4p;
create table variables_gcs_eicu_v4p as
select c.patientunitstayid,
    min(case when nc.nursingchartvalue ~ '^\d+$'
             then cast(nc.nursingchartvalue as numeric) end) as gcs_min
from dataset_modelo_eicu_v4p c
left join "nurseCharting" nc on c.patientunitstayid = nc.patientunitstayid
    and nc.nursingchartoffset between 0 and 180
    and nc.nursingchartcelltypevalname = 'GCS Total'
group by c.patientunitstayid;


-- 6. SOFA (ventana 0-3h) — variable del modelo
drop table if exists variables_sofa_eicu_v4p;
create table variables_sofa_eicu_v4p as
with vasopresores as (
    select c.patientunitstayid,
        max(case when lower(id.drugname) like '%norepinephrine%'
                   or lower(id.drugname) like '%levophed%'          then 1 else 0 end) as con_norad,
        max(case when lower(id.drugname) like '%epinephrine%'
                  and lower(id.drugname) not like '%norepinephrine%' then 1 else 0 end) as con_epi,
        max(case when lower(id.drugname) like '%dopamine%'          then 1 else 0 end) as con_dopa,
        max(case when lower(id.drugname) like '%dobutamine%'        then 1 else 0 end) as con_dobu
    from dataset_modelo_eicu_v4p c
    left join "infusionDrug" id on c.patientunitstayid = id.patientunitstayid
        and id.infusionoffset between 0 and 180
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
    from dataset_modelo_eicu_v4p c
    left join variables_lab_eicu_v4p     l  on c.patientunitstayid = l.patientunitstayid
    left join variables_fio2_eicu_v4p    f  on c.patientunitstayid = f.patientunitstayid
    left join variables_gcs_eicu_v4p     g  on c.patientunitstayid = g.patientunitstayid
    left join variables_vitales_eicu_v4p vt on c.patientunitstayid = vt.patientunitstayid
    left join vasopresores               v  on c.patientunitstayid = v.patientunitstayid
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


-- 7. TABLA FINAL — identificadores + etiqueta + 4 variables del modelo únicamente
drop table if exists dataset_final_eicu_v4p;
create table dataset_final_eicu_v4p as
select
    c.uniquepid              as subject_id,
    c.patientunitstayid      as stay_id,
    c.anchor_age,
    c.gender,
    c.peso_kg,
    c.contador_estancia_uci,
    c.horas_hasta_norad,
    c.etiqueta_norad_3_12,
    vt.map_min,
    case
        when l.pao2_min is not null and f.fio2_max is not null and f.fio2_max > 0
        then l.pao2_min / (f.fio2_max / 100.0)
        else null
    end                      as pf_min,
    so.sofa_max,
    l.tp_max
from dataset_modelo_eicu_v4p     c
left join variables_lab_eicu_v4p     l  on c.patientunitstayid = l.patientunitstayid
left join variables_vitales_eicu_v4p vt on c.patientunitstayid = vt.patientunitstayid
left join variables_fio2_eicu_v4p    f  on c.patientunitstayid = f.patientunitstayid
left join variables_sofa_eicu_v4p    so on c.patientunitstayid = so.patientunitstayid;


-- 8. TABLA FINAL LIMPIA — solo filas completas en las 4 variables del modelo
drop table if exists dataset_final_eicu_v4p_clean;
create table dataset_final_eicu_v4p_clean as
select * from dataset_final_eicu_v4p
where map_min  is not null
  and pf_min   is not null
  and sofa_max is not null
  and tp_max   is not null;


-- VERIFICACIÓN
select
    count(*)                                               as total_filas,
    count(distinct subject_id)                             as pacientes_unicos,
    sum(etiqueta_norad_3_12)                               as positivos,
    round(100.0 * sum(etiqueta_norad_3_12) / count(*), 2) as prevalencia_pct
from dataset_final_eicu_v4p_clean;


-- DESCARGAR
select * from dataset_final_eicu_v4p_clean;