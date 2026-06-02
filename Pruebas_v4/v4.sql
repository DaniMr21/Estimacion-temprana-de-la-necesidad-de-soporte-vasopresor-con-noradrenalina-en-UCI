
-- 1. NORADRENALINA: primer inicio por estancia

drop table if exists public.noradrenalina_v4;

create table public.noradrenalina_v4 as
select
    stay_id,
    min(starttime) as inicio_norad
from mimiciv_icu.inputevents
where itemid = 221906          -- Noradrenalina
  and amount > 0
group by stay_id;



-- 2. COHORTE BASE: adultos con estancia UCI > 24h

drop table if exists public.cohorte_base_v4;

create table public.cohorte_base_v4 as
select
    i.subject_id,
    i.hadm_id,
    i.stay_id,
    i.intime,
    i.outtime,
    p.anchor_age,
    p.gender,
    row_number() over (partition by i.subject_id order by i.intime) as contador_estancia_uci
from mimiciv_icu.icustays i
join mimiciv_hosp.patients p on i.subject_id = p.subject_id
where p.anchor_age >= 18
  and extract(epoch from (i.outtime - i.intime)) / 3600.0 > 24;

-- 3. COHORTE + NORAD: horas hasta el primer inicio (NULL si nunca recibe)

drop table if exists public.cohorte_norad_v4;

create table public.cohorte_norad_v4 as
select
    c.subject_id,
    c.hadm_id,
    c.stay_id,
    c.intime,
    c.outtime,
    c.anchor_age,
    c.gender,
    c.contador_estancia_uci,
    n.inicio_norad,
    extract(epoch from (n.inicio_norad - c.intime)) / 3600.0 as horas_hasta_norad
from public.cohorte_base_v4 c
left join public.noradrenalina_v4 n on c.stay_id = n.stay_id;


drop table if exists public.cohorte_norad_v4_limpio;

create table public.cohorte_norad_v4_limpio as
select *
from public.cohorte_norad_v4
where horas_hasta_norad is null
   or horas_hasta_norad >= 0;


-- 4. DATASET MODELO: etiqueta + exclusión de norad en ventana de observación

drop table if exists public.dataset_modelo_v4;

create table public.dataset_modelo_v4 as
select *,
    case
        when horas_hasta_norad between 6 and 24 then 1
        else 0
    end as etiqueta_norad_6_24
from public.cohorte_norad_v4_limpio
where horas_hasta_norad is null
   or horas_hasta_norad >= 6;


-- 5. VARIABLES DE LABORATORIO (ventana 0-6h)

drop table if exists public.variables_lab_v4;

create table public.variables_lab_v4 as
select
    c.stay_id,

    -- Lactato
    avg(case when l.itemid = 50813 then l.valuenum end) as lactato_media,
    min(case when l.itemid = 50813 then l.valuenum end) as lactato_min,
    max(case when l.itemid = 50813 then l.valuenum end) as lactato_max,

    -- Creatinina
    avg(case when l.itemid = 50912 then l.valuenum end) as creatinina_media,
    min(case when l.itemid = 50912 then l.valuenum end) as creatinina_min,
    max(case when l.itemid = 50912 then l.valuenum end) as creatinina_max,

    -- Plaquetas
    avg(case when l.itemid = 51265 then l.valuenum end) as plaquetas_media,
    min(case when l.itemid = 51265 then l.valuenum end) as plaquetas_min,
    max(case when l.itemid = 51265 then l.valuenum end) as plaquetas_max,

    -- Bilirrubina
    avg(case when l.itemid = 50885 then l.valuenum end) as bilirrubina_media,
    min(case when l.itemid = 50885 then l.valuenum end) as bilirrubina_min,
    max(case when l.itemid = 50885 then l.valuenum end) as bilirrubina_max,

    -- TP
    avg(case when l.itemid = 51274 then l.valuenum end) as tp_media,
    min(case when l.itemid = 51274 then l.valuenum end) as tp_min,
    max(case when l.itemid = 51274 then l.valuenum end) as tp_max,

    -- GPT (ALT)
    avg(case when l.itemid = 50861 then l.valuenum end) as gpt_media,
    min(case when l.itemid = 50861 then l.valuenum end) as gpt_min,
    max(case when l.itemid = 50861 then l.valuenum end) as gpt_max,

    -- GOT (AST)
    avg(case when l.itemid = 50878 then l.valuenum end) as got_media,
    min(case when l.itemid = 50878 then l.valuenum end) as got_min,
    max(case when l.itemid = 50878 then l.valuenum end) as got_max,

    -- PaO2
    avg(case when l.itemid = 50821 then l.valuenum end) as pao2_media,
    min(case when l.itemid = 50821 then l.valuenum end) as pao2_min,
    max(case when l.itemid = 50821 then l.valuenum end) as pao2_max,

    -- pH
    avg(case when l.itemid = 50820 then l.valuenum end) as ph_media,
    min(case when l.itemid = 50820 then l.valuenum end) as ph_min,
    max(case when l.itemid = 50820 then l.valuenum end) as ph_max,

    -- Neutrófilos
    avg(case when l.itemid = 51256 then l.valuenum end) as neutro_media,
    min(case when l.itemid = 51256 then l.valuenum end) as neutro_min,
    max(case when l.itemid = 51256 then l.valuenum end) as neutro_max,

    -- Leucocitos
    avg(case when l.itemid = 51301 then l.valuenum end) as leucocitos_media,
    min(case when l.itemid = 51301 then l.valuenum end) as leucocitos_min,
    max(case when l.itemid = 51301 then l.valuenum end) as leucocitos_max,

    -- PaCO2
    avg(case when l.itemid = 50818 then l.valuenum end) as paco2_media,
    min(case when l.itemid = 50818 then l.valuenum end) as paco2_min,
    max(case when l.itemid = 50818 then l.valuenum end) as paco2_max,

    -- Bicarbonato
    avg(case when l.itemid = 50882 then l.valuenum end) as bicarbonato_media,
    min(case when l.itemid = 50882 then l.valuenum end) as bicarbonato_min,
    max(case when l.itemid = 50882 then l.valuenum end) as bicarbonato_max,

    -- Glucemia
    avg(case when l.itemid = 50931 then l.valuenum end) as glucemia_media,
    min(case when l.itemid = 50931 then l.valuenum end) as glucemia_min,
    max(case when l.itemid = 50931 then l.valuenum end) as glucemia_max,

    -- Sodio
    avg(case when l.itemid = 50824 then l.valuenum end) as sodio_media,
    min(case when l.itemid = 50824 then l.valuenum end) as sodio_min,
    max(case when l.itemid = 50824 then l.valuenum end) as sodio_max,

    -- Potasio
    avg(case when l.itemid = 50822 then l.valuenum end) as potasio_media,
    min(case when l.itemid = 50822 then l.valuenum end) as potasio_min,
    max(case when l.itemid = 50822 then l.valuenum end) as potasio_max,

    -- Hemoglobina
    avg(case when l.itemid = 51222 then l.valuenum end) as hemoglobina_media,
    min(case when l.itemid = 51222 then l.valuenum end) as hemoglobina_min,
    max(case when l.itemid = 51222 then l.valuenum end) as hemoglobina_max

from public.dataset_modelo_v4 c
left join mimiciv_hosp.labevents l
    on c.hadm_id = l.hadm_id
    and l.valuenum is not null
    and l.charttime >= c.intime
    and l.charttime <= c.intime + interval '6 hour'
    and l.itemid in (
        50813, 50912, 51265, 50885, 51274, 50861, 50878,
        50821, 50820, 51256, 51301, 50818, 50882, 50931,
        50824, 50822, 51222
    )
group by c.stay_id;



-- 6. CONSTANTES VITALES (ventana 0-6h)

drop table if exists public.variables_chart_v4;

create table public.variables_chart_v4 as
select
    c.stay_id,

    -- Frecuencia cardíaca
    avg(case when ch.itemid = 220045 then ch.valuenum end) as hr_media,
    min(case when ch.itemid = 220045 then ch.valuenum end) as hr_min,
    max(case when ch.itemid = 220045 then ch.valuenum end) as hr_max,

    -- Frecuencia respiratoria
    avg(case when ch.itemid = 220210 then ch.valuenum end) as rr_media,
    min(case when ch.itemid = 220210 then ch.valuenum end) as rr_min,
    max(case when ch.itemid = 220210 then ch.valuenum end) as rr_max,

    -- Temperatura (223762 en °C, 223761 en °F con conversión)
    avg(case
        when ch.itemid = 223762 then ch.valuenum
        when ch.itemid = 223761 then (ch.valuenum - 32) * 5.0 / 9.0
    end) as temp_media,
    min(case
        when ch.itemid = 223762 then ch.valuenum
        when ch.itemid = 223761 then (ch.valuenum - 32) * 5.0 / 9.0
    end) as temp_min,
    max(case
        when ch.itemid = 223762 then ch.valuenum
        when ch.itemid = 223761 then (ch.valuenum - 32) * 5.0 / 9.0
    end) as temp_max,

    -- SpO2
    avg(case when ch.itemid = 220277 then ch.valuenum end) as spo2_media,
    min(case when ch.itemid = 220277 then ch.valuenum end) as spo2_min,
    max(case when ch.itemid = 220277 then ch.valuenum end) as spo2_max,

    -- MAP arterial
    avg(case when ch.itemid = 220052 then ch.valuenum end) as map_art_media,
    min(case when ch.itemid = 220052 then ch.valuenum end) as map_art_min,
    max(case when ch.itemid = 220052 then ch.valuenum end) as map_art_max,

    -- MAP NIBP
    avg(case when ch.itemid = 220181 then ch.valuenum end) as map_nibp_media,
    min(case when ch.itemid = 220181 then ch.valuenum end) as map_nibp_min,
    max(case when ch.itemid = 220181 then ch.valuenum end) as map_nibp_max,

    -- FiO2
    avg(case when ch.itemid = 223835 then ch.valuenum end) as fio2_media,
    min(case when ch.itemid = 223835 then ch.valuenum end) as fio2_min,
    max(case when ch.itemid = 223835 then ch.valuenum end) as fio2_max

from public.dataset_modelo_v4 c
left join mimiciv_icu.chartevents ch
    on c.stay_id = ch.stay_id
    and ch.valuenum is not null
    and ch.charttime >= c.intime
    and ch.charttime <= c.intime + interval '6 hour'
    and ch.itemid in (220045, 220210, 223762, 223761, 220277, 220052, 220181, 223835)
group by c.stay_id;


-- 7. GCS (suma de los 3 componentes) — ventana 0-6h

drop table if exists public.variables_gcs_v4;

create table public.variables_gcs_v4 as
with gcs_componentes as (
    select
        c.stay_id,
        ch.charttime,
        max(case when ch.itemid = 220739 then ch.valuenum end) as gcs_eye,
        max(case when ch.itemid = 223900 then ch.valuenum end) as gcs_verbal,
        max(case when ch.itemid = 223901 then ch.valuenum end) as gcs_motor
    from public.dataset_modelo_v4 c
    left join mimiciv_icu.chartevents ch
        on c.stay_id = ch.stay_id
        and ch.valuenum is not null
        and ch.charttime >= c.intime
        and ch.charttime <= c.intime + interval '6 hour'
        and ch.itemid in (220739, 223900, 223901)
    group by c.stay_id, ch.charttime
),
gcs_total as (
    select
        stay_id,
        charttime,
        (gcs_eye + gcs_verbal + gcs_motor) as gcs_total
    from gcs_componentes
    where gcs_eye    is not null
      and gcs_verbal is not null
      and gcs_motor  is not null
)
select
    stay_id,
    avg(gcs_total) as gcs_media,
    min(gcs_total) as gcs_min,
    max(gcs_total) as gcs_max
from gcs_total
group by stay_id;



-- 8. DIURESIS (ventana 0-6h)

drop table if exists public.variables_diuresis_v4;

create table public.variables_diuresis_v4 as
with cohorte as (
    select c.stay_id, c.intime
    from public.dataset_modelo_v4 c
),
volumen_6h as (
    select
        c.stay_id,
        sum(uo.urineoutput) as diuresis_volumen_6h,
        count(uo.urineoutput) as diuresis_n_registros
    from cohorte c
    left join mimiciv_derived.urine_output uo
        on c.stay_id = uo.stay_id
        and uo.charttime >= c.intime
        and uo.charttime <= c.intime + interval '6 hour'
    group by c.stay_id
),
peso_paciente as (
    select distinct on (c.stay_id)
        c.stay_id,
        wd.weight
    from cohorte c
    left join mimiciv_derived.weight_durations wd
        on c.stay_id = wd.stay_id
        and wd.starttime <= c.intime + interval '6 hour'
        and wd.endtime   >= c.intime
        and wd.weight > 0
    order by c.stay_id, wd.starttime
)
select
    v.stay_id,
    v.diuresis_volumen_6h,
    p.weight as peso_kg,
    case
        when p.weight > 0 and v.diuresis_volumen_6h is not null
        then v.diuresis_volumen_6h / p.weight
        else null
    end as diuresis_ml_kg_6h
from volumen_6h v
left join peso_paciente p on v.stay_id = p.stay_id;


-- 9. SEPSIS-3 reconocida en la ventana 0-6h

-- Fuente: mimiciv_derived.sepsis3 (repo oficial MIT-LCP/mimic-code).
drop table if exists public.variables_sepsis_v4;

create table public.variables_sepsis_v4 as
select
    c.stay_id,
    max(
        case
            when s.sepsis3 is true
             and s.suspected_infection_time >= c.intime
             and s.suspected_infection_time <= c.intime + interval '6 hour'
             and s.sofa_time                <= c.intime + interval '6 hour'
            then 1
            else 0
        end
    ) as tiene_sepsis
from public.dataset_modelo_v4 c
left join mimiciv_derived.sepsis3 s
    on c.stay_id = s.stay_id
group by c.stay_id;



-- 10. SOFA score en la ventana 0-6h  

-- Fuente: mimiciv_derived.sofa (repo oficial MIT-LCP/mimic-code).
-- La tabla produce una fila por hora por estancia con `sofa_24hours`, que es
-- la suma de los 6 subscores calculada sobre la ventana de 24h previas al `endtime` de esa fila. Nos quedamos con las filas cuyo endtime cae dentro
-- de primeras 6h desde intime y agregamos media/min/max del SOFA total.

drop table if exists public.variables_sofa_v4;

create table public.variables_sofa_v4 as
select
    c.stay_id,
    avg(s.sofa_24hours) as sofa_media,
    min(s.sofa_24hours) as sofa_min,
    max(s.sofa_24hours) as sofa_max
from public.dataset_modelo_v4 c
left join mimiciv_derived.sofa s
    on c.stay_id = s.stay_id
    and s.endtime >= c.intime
    and s.endtime <= c.intime + interval '6 hour'
group by c.stay_id;


-- -----------------------------------------------------------------------------
-- 11. VENTILACIÓN INVASIVA solapada con la ventana 0-6h  
-- -----------------------------------------------------------------------------
-- Fuente: mimiciv_derived.ventilation (repo oficial MIT-LCP/mimic-code).
-- Marca 1 si existe algún tramo de ventilación invasiva o traqueostomía que
-- solape (aunque sea parcialmente) con la ventana [intime, intime + 6h].
drop table if exists public.variables_ventilacion_v4;

create table public.variables_ventilacion_v4 as
select
    c.stay_id,
    case when exists (
        select 1
        from mimiciv_derived.ventilation v
        where v.stay_id = c.stay_id
          and v.starttime <= c.intime + interval '6 hour'
          and v.endtime   >= c.intime
          and v.ventilation_status in ('InvasiveVent', 'Tracheostomy', 'Trach')
    ) then 1 else 0 end as ventilacion_invasiva_6h
from public.dataset_modelo_v4 c;



-- 12. TABLA FINAL v4 (unión + variables derivadas)


drop table if exists public.dataset_final_v4;

create table public.dataset_final_v4 as
select
    -- Identificadores y contexto
    c.subject_id,
    c.hadm_id,
    c.stay_id,
    c.anchor_age,
    c.gender,
    c.contador_estancia_uci,
    c.horas_hasta_norad,
    c.etiqueta_norad_6_24,

    -- Sepsis, SOFA y ventilación
    coalesce(s.tiene_sepsis, 0)             as tiene_sepsis,
    so.sofa_media, so.sofa_min, so.sofa_max,
    coalesce(vm.ventilacion_invasiva_6h, 0) as ventilacion_invasiva_6h,

    -- Laboratorio
    l.lactato_media,     l.lactato_min,     l.lactato_max,
    l.creatinina_media,  l.creatinina_min,  l.creatinina_max,
    l.plaquetas_media,   l.plaquetas_min,   l.plaquetas_max,
    l.bilirrubina_media, l.bilirrubina_min, l.bilirrubina_max,
    l.tp_media,          l.tp_min,          l.tp_max,
    l.gpt_media,         l.gpt_min,         l.gpt_max,
    l.got_media,         l.got_min,         l.got_max,
    l.pao2_media,        l.pao2_min,        l.pao2_max,
    l.ph_media,          l.ph_min,          l.ph_max,
    l.neutro_media,      l.neutro_min,      l.neutro_max,
    l.leucocitos_media,  l.leucocitos_min,  l.leucocitos_max,
    l.paco2_media,       l.paco2_min,       l.paco2_max,
    l.bicarbonato_media, l.bicarbonato_min, l.bicarbonato_max,
    l.glucemia_media,    l.glucemia_min,    l.glucemia_max,
    l.sodio_media,       l.sodio_min,       l.sodio_max,
    l.potasio_media,     l.potasio_min,     l.potasio_max,
    l.hemoglobina_media, l.hemoglobina_min, l.hemoglobina_max,

    -- Vitales
    ch.hr_media,   ch.hr_min,   ch.hr_max,
    ch.rr_media,   ch.rr_min,   ch.rr_max,
    ch.temp_media, ch.temp_min, ch.temp_max,
    ch.spo2_media, ch.spo2_min, ch.spo2_max,
    ch.fio2_media, ch.fio2_min, ch.fio2_max,

    -- MAP combinada (prioriza arterial sobre NIBP)
    coalesce(ch.map_art_media, ch.map_nibp_media) as map_media,
    coalesce(ch.map_art_min,   ch.map_nibp_min)   as map_min,
    coalesce(ch.map_art_max,   ch.map_nibp_max)   as map_max,

    -- Índice P/F (PaO2 / FiO2 como fracción), trío media/min/max
    case
        when l.pao2_media is not null and ch.fio2_media is not null and ch.fio2_media > 0
        then l.pao2_media / (ch.fio2_media / 100.0)
        else null
    end as pf_media,
    case
        when l.pao2_min is not null and ch.fio2_max is not null and ch.fio2_max > 0
        then l.pao2_min / (ch.fio2_max / 100.0)
        else null
    end as pf_min,
    case
        when l.pao2_max is not null and ch.fio2_min is not null and ch.fio2_min > 0
        then l.pao2_max / (ch.fio2_min / 100.0)
        else null
    end as pf_max,

    -- GCS
    g.gcs_media, g.gcs_min, g.gcs_max,

    -- Diuresis
    d.diuresis_volumen_6h,
    d.peso_kg,
    d.diuresis_ml_kg_6h

from public.dataset_modelo_v4 c
left join public.variables_lab_v4         l  on c.stay_id = l.stay_id
left join public.variables_chart_v4       ch on c.stay_id = ch.stay_id
left join public.variables_gcs_v4         g  on c.stay_id = g.stay_id
left join public.variables_diuresis_v4    d  on c.stay_id = d.stay_id
left join public.variables_sepsis_v4      s  on c.stay_id = s.stay_id
left join public.variables_sofa_v4        so on c.stay_id = so.stay_id
left join public.variables_ventilacion_v4 vm on c.stay_id = vm.stay_id;


-- -----------------------------------------------------------------------------
-- 13. TABLA FINAL LIMPIA (sin missings en predictores clave)
-- -----------------------------------------------------------------------------
-- `tiene_sepsis` y `ventilacion_invasiva_6h` NO se filtran por is not null: son binarias resueltas a 0/1 por construcción.
-- SOFA y P/F SÍ se filtran para evitar NaN en los modelos.
drop table if exists public.dataset_final_v4_clean;

create table public.dataset_final_v4_clean as
select
    subject_id,
    hadm_id,
    stay_id,
    anchor_age,
    gender,
    contador_estancia_uci,
    horas_hasta_norad,
    etiqueta_norad_6_24,
    tiene_sepsis,
    sofa_media, sofa_min, sofa_max,
    ventilacion_invasiva_6h,
    lactato_media,     lactato_min,     lactato_max,
    creatinina_media,  creatinina_min,  creatinina_max,
    plaquetas_media,   plaquetas_min,   plaquetas_max,
    bilirrubina_media, bilirrubina_min, bilirrubina_max,
    tp_media,          tp_min,          tp_max,
    gpt_media,         gpt_min,         gpt_max,
    got_media,         got_min,         got_max,
    pao2_media,        pao2_min,        pao2_max,
    ph_media,          ph_min,          ph_max,
    leucocitos_media,  leucocitos_min,  leucocitos_max,
    paco2_media,       paco2_min,       paco2_max,
    bicarbonato_media, bicarbonato_min, bicarbonato_max,
    glucemia_media,    glucemia_min,    glucemia_max,
    hemoglobina_media, hemoglobina_min, hemoglobina_max,
    hr_media,   hr_min,   hr_max,
    rr_media,   rr_min,   rr_max,
    temp_media, temp_min, temp_max,
    spo2_media, spo2_min, spo2_max,
    map_media,  map_min,  map_max,
    fio2_media, fio2_min, fio2_max,
    pf_media,   pf_min,   pf_max,
    gcs_media,  gcs_min,  gcs_max,
    diuresis_volumen_6h, peso_kg, diuresis_ml_kg_6h
from public.dataset_final_v4
where lactato_media     is not null
  and creatinina_media  is not null
  and plaquetas_media   is not null
  and bilirrubina_media is not null
  and tp_media          is not null
  and gpt_media         is not null
  and got_media         is not null
  and pao2_media        is not null
  and ph_media          is not null
  and leucocitos_media  is not null
  and paco2_media       is not null
  and bicarbonato_media is not null
  and glucemia_media    is not null
  and hemoglobina_media is not null
  and hr_media          is not null
  and rr_media          is not null
  and temp_media        is not null
  and spo2_media        is not null
  and map_media         is not null
  and fio2_media        is not null
  and gcs_media         is not null
  and diuresis_ml_kg_6h is not null
  and sofa_media        is not null
  and pf_media          is not null;

-- VERIFICACIONES

-- Resumen global
select
    count(*)                                              as total_filas,
    count(distinct subject_id)                            as pacientes_unicos,
    sum(etiqueta_norad_6_24)                              as positivos,
    round(100.0 * sum(etiqueta_norad_6_24) / count(*), 2) as prevalencia_pct,
    sum(tiene_sepsis)                                     as con_sepsis_6h,
    round(100.0 * sum(tiene_sepsis) / count(*), 2)        as pct_sepsis_6h,
    sum(ventilacion_invasiva_6h)                          as con_vmi_6h,
    round(100.0 * sum(ventilacion_invasiva_6h) / count(*), 2) as pct_vmi_6h,
    round(avg(sofa_media)::numeric, 2)                    as sofa_medio
from public.dataset_final_v4_clean;

select * from public.dataset_final_v4_clean;