-- =============================================================================
-- DATASET VALIDACIÓN EXTERNA eICU — ventana 6-24h
-- Modelo: XGBoost Medio_6_24 (calibrado Platt)
-- Variables del modelo: pf_min, map_min, diuresis_ml_kg_6h,
--                       hr_media, sofa_max, ventilacion_invasiva_6h
-- =============================================================================


-- 1. NORADRENALINA: primer inicio por estancia
drop table if exists noradrenalina_eicu;
create table noradrenalina_eicu as
select patientunitstayid, min(infusionoffset) as inicio_norad_min
from "infusionDrug"
where (lower(drugname) like '%norepinephrine%' or lower(drugname) like '%levophed%')
  and lower(drugname) not like '%volume%'
group by patientunitstayid;


-- 2. COHORTE BASE: adultos ≥18, estancia UCI > 24h
drop table if exists cohorte_base_eicu;
create table cohorte_base_eicu as
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
  and p.unitdischargeoffset > 1440;


-- 3. COHORTE + NORAD
drop table if exists cohorte_norad_eicu;
create table cohorte_norad_eicu as
select c.*, n.inicio_norad_min, n.inicio_norad_min / 60.0 as horas_hasta_norad
from cohorte_base_eicu c
left join noradrenalina_eicu n on c.patientunitstayid = n.patientunitstayid
where n.inicio_norad_min is null or n.inicio_norad_min >= 0;


-- 4. DATASET MODELO v4: etiqueta + exclusión norad <6h
drop table if exists dataset_modelo_eicu_v4;
create table dataset_modelo_eicu_v4 as
select *,
    case
        when horas_hasta_norad between 6 and 24 then 1
        else 0
    end as etiqueta_norad_6_24
from cohorte_norad_eicu
where horas_hasta_norad is null or horas_hasta_norad >= 6;


-- 5. LABORATORIO (ventana 0-6h)
-- pao2_min: para calcular pf_min
-- bilirrubina, plaquetas, creatinina: auxiliares para SOFA interno únicamente
drop table if exists variables_lab_eicu_v4;
create table variables_lab_eicu_v4 as
select c.patientunitstayid,
    min(case when l.labname = 'paO2'             then l.labresult end) as pao2_min,
    max(case when l.labname = 'total bilirubin'  then l.labresult end) as bilirrubina_max,
    min(case when l.labname = 'platelets x 1000' then l.labresult end) as plaquetas_min,
    max(case when l.labname = 'creatinine'       then l.labresult end) as creatinina_max
from dataset_modelo_eicu_v4 c
left join "lab" l on c.patientunitstayid = l.patientunitstayid
    and l.labresult is not null
    and l.labresultoffset between 0 and 360
    and l.labname in ('paO2', 'total bilirubin', 'platelets x 1000', 'creatinine')
group by c.patientunitstayid;


-- 6. VITALES (ventana 0-6h) — solo map_min y hr_media
drop table if exists variables_vitales_eicu_v4;
create table variables_vitales_eicu_v4 as
with periodic as (
    select c.patientunitstayid,
        avg(vp.heartrate)    as hr_media,
        min(vp.systemicmean) as map_art_min
    from dataset_modelo_eicu_v4 c
    left join "vitalPeriodic" vp on c.patientunitstayid = vp.patientunitstayid
        and vp.observationoffset between 0 and 360
    group by c.patientunitstayid
),
aperiodic as (
    select c.patientunitstayid,
        min(va.noninvasivemean) as map_nibp_min
    from dataset_modelo_eicu_v4 c
    left join "vitalAperiodic" va on c.patientunitstayid = va.patientunitstayid
        and va.observationoffset between 0 and 360
    group by c.patientunitstayid
)
select p.patientunitstayid,
    p.hr_media,
    coalesce(p.map_art_min, a.map_nibp_min) as map_min
from periodic p
left join aperiodic a on p.patientunitstayid = a.patientunitstayid;


-- 7. FiO2 NORMALIZADO (ventana 0-6h) — para pf_min y SOFA respiratorio
drop table if exists variables_fio2_eicu_v4;
create table variables_fio2_eicu_v4 as
with raw as (
    select c.patientunitstayid,
        case when rc.respchartvalue ~ '^\d+(\.\d+)?$'
             then cast(rc.respchartvalue as numeric) end as fio2_raw
    from dataset_modelo_eicu_v4 c
    left join "respiratoryCharting" rc on c.patientunitstayid = rc.patientunitstayid
        and rc.respchartoffset between 0 and 360
        and rc.respchartvaluelabel in ('FiO2', 'FIO2 (%)', 'Set Fraction of Inspired Oxygen (FIO2)')
),
normalizado as (
    select patientunitstayid,
        case
            when fio2_raw between 0.21 and 1.0 then fio2_raw * 100
            else fio2_raw
        end as fio2_pct
    from raw
)
select patientunitstayid, max(fio2_pct) as fio2_max
from normalizado where fio2_pct is not null
group by patientunitstayid;


-- 8. GCS (ventana 0-6h) — solo para SOFA neurológico
drop table if exists variables_gcs_eicu_v4;
create table variables_gcs_eicu_v4 as
select c.patientunitstayid,
    min(case when nc.nursingchartvalue ~ '^\d+$'
             then cast(nc.nursingchartvalue as numeric) end) as gcs_min
from dataset_modelo_eicu_v4 c
left join "nurseCharting" nc on c.patientunitstayid = nc.patientunitstayid
    and nc.nursingchartoffset between 0 and 360
    and nc.nursingchartcelltypevalname = 'GCS Total'
group by c.patientunitstayid;


-- 9. DIURESIS normalizada por peso (ventana 0-6h) — variable del modelo
drop table if exists variables_diuresis_eicu_v4;
create table variables_diuresis_eicu_v4 as
select c.patientunitstayid,
    case
        when c.peso_kg > 0 and c.peso_kg is not null
        then sum(case
            when io.celllabel in (
                'Urine','URINE CATHETER','Voided Amount','Foley cath','Foley',
                'Urine, void:','foley','Urine Output-Foley','SN Urine Output(ml)',
                'Urine Output (mL)-Urethral Catheter','Urine Output-foley',
                'Urine Output (mL)-Urethral Catheter ',
                'Urine Output (mL)-Urethral Catheter  ','Urine Output-FOLEY',
                'FOLEY','Urine Output-Urine Output')
              and io.cellvaluenumeric > 0
            then io.cellvaluenumeric
            else 0
        end) / c.peso_kg
        else null
    end as diuresis_ml_kg_6h
from dataset_modelo_eicu_v4 c
left join "intakeOutput" io on c.patientunitstayid = io.patientunitstayid
    and io.intakeoutputoffset between 0 and 360
group by c.patientunitstayid, c.peso_kg;


-- 10. VENTILACIÓN INVASIVA (ventana 0-6h) — variable del modelo
-- Se detecta por presencia de registros de ventilador en respiratoryCharting
-- AVISO: verificar que estos labels existen en tu instancia eICU
-- Si el paciente tiene ≥1 registro de parámetros de ventilador → ventilacion_invasiva_6h = 1
drop table if exists variables_ventilacion_eicu_v4;
create table variables_ventilacion_eicu_v4 as
select c.patientunitstayid,
    max(case when rc.respchartvaluelabel in (
        'Vent Rate',
        'Tidal Volume (set)',
        'Tidal Volume (observed)',
        'TV/kg IBW',
        'PEEP',
        'Peak Insp. Pressure',
        'Plateau Pressure',
        'Mean Airway Pressure',
        'Inspiratory Ratio',
        'Inspiratory Time'
    ) then 1 else 0 end) as ventilacion_invasiva_6h
from dataset_modelo_eicu_v4 c
left join "respiratoryCharting" rc on c.patientunitstayid = rc.patientunitstayid
    and rc.respchartoffset between 0 and 360
group by c.patientunitstayid;


-- 11. SOFA (ventana 0-6h) — variable del modelo
drop table if exists variables_sofa_eicu_v4;
create table variables_sofa_eicu_v4 as
with vasopresores as (
    select c.patientunitstayid,
        max(case when lower(id.drugname) like '%norepinephrine%'
                   or lower(id.drugname) like '%levophed%'          then 1 else 0 end) as con_norad,
        max(case when lower(id.drugname) like '%epinephrine%'
                  and lower(id.drugname) not like '%norepinephrine%' then 1 else 0 end) as con_epi,
        max(case when lower(id.drugname) like '%dopamine%'          then 1 else 0 end) as con_dopa,
        max(case when lower(id.drugname) like '%dobutamine%'        then 1 else 0 end) as con_dobu
    from dataset_modelo_eicu_v4 c
    left join "infusionDrug" id on c.patientunitstayid = id.patientunitstayid
        and id.infusionoffset between 0 and 360
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
    from dataset_modelo_eicu_v4 c
    left join variables_lab_eicu_v4     l  on c.patientunitstayid = l.patientunitstayid
    left join variables_fio2_eicu_v4    f  on c.patientunitstayid = f.patientunitstayid
    left join variables_gcs_eicu_v4     g  on c.patientunitstayid = g.patientunitstayid
    left join variables_vitales_eicu_v4 vt on c.patientunitstayid = vt.patientunitstayid
    left join vasopresores              v  on c.patientunitstayid = v.patientunitstayid
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


-- 12. TABLA FINAL — identificadores + etiqueta + 6 variables del modelo
drop table if exists dataset_final_eicu_v4;
create table dataset_final_eicu_v4 as
select
    c.uniquepid              as subject_id,
    c.patientunitstayid      as stay_id,
    c.anchor_age,
    c.gender,
    c.peso_kg,
    c.contador_estancia_uci,
    c.horas_hasta_norad,
    c.etiqueta_norad_6_24,
    -- 6 variables del modelo XGB Medio
    case
        when l.pao2_min is not null and f.fio2_max is not null and f.fio2_max > 0
        then l.pao2_min / (f.fio2_max / 100.0)
        else null
    end                      as pf_min,
    vt.map_min,
    d.diuresis_ml_kg_6h,
    vt.hr_media,
    so.sofa_max,
    vc.ventilacion_invasiva_6h
from dataset_modelo_eicu_v4      c
left join variables_lab_eicu_v4      l  on c.patientunitstayid = l.patientunitstayid
left join variables_vitales_eicu_v4  vt on c.patientunitstayid = vt.patientunitstayid
left join variables_fio2_eicu_v4     f  on c.patientunitstayid = f.patientunitstayid
left join variables_diuresis_eicu_v4 d  on c.patientunitstayid = d.patientunitstayid
left join variables_sofa_eicu_v4     so on c.patientunitstayid = so.patientunitstayid
left join variables_ventilacion_eicu_v4 vc on c.patientunitstayid = vc.patientunitstayid;


-- 13. TABLA FINAL LIMPIA — solo filas completas en las 6 variables del modelo
drop table if exists dataset_final_eicu_v4_clean;
create table dataset_final_eicu_v4_clean as
select * from dataset_final_eicu_v4
where pf_min               is not null
  and map_min              is not null
  and diuresis_ml_kg_6h   is not null
  and hr_media             is not null
  and sofa_max             is not null
  and ventilacion_invasiva_6h is not null;


-- VERIFICACIÓN
select
    count(*)                                               as total_filas,
    count(distinct subject_id)                             as pacientes_unicos,
    sum(etiqueta_norad_6_24)                               as positivos,
    round(100.0 * sum(etiqueta_norad_6_24) / count(*), 2) as prevalencia_pct
from dataset_final_eicu_v4_clean;


-- DESCARGAR
select * from dataset_final_eicu_v4_clean;
