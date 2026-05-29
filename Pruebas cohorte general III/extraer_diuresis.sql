--AVISO: ESTE SCRIPT SE OBTUVO DIRECTAMENTE A TRAVÉS DEL REPOSITORIO DEL MIT-LCP PARA TRABAJAR CON MIMIC-IV: https://github.com/MIT-LCP/mimic-code


-- THIS SCRIPT IS AUTOMATICALLY GENERATED. DO NOT EDIT IT DIRECTLY.
DROP TABLE IF EXISTS mimiciv_derived.weight_durations; CREATE TABLE mimiciv_derived.weight_durations AS
/* This query extracts weights for adult ICU patients with start/stop times */ /* if an admission weight is given, then this is assigned from intime to outtime */
WITH wt_stg AS (
  SELECT
    c.stay_id,
    c.charttime,
    CASE WHEN c.itemid = 226512 THEN 'admit' ELSE 'daily' END AS weight_type, /* TODO: eliminate obvious outliers if there is a reasonable weight */
    c.valuenum AS weight
  FROM mimiciv_icu.chartevents AS c
  WHERE
    NOT c.valuenum IS NULL
    AND c.itemid IN (226512 /* Admit Wt */, 224639 /* Daily Weight */)
    AND c.valuenum > 0
), wt_stg1 AS (
  SELECT
    stay_id,
    charttime,
    weight_type,
    weight,
    ROW_NUMBER() OVER (PARTITION BY stay_id, weight_type ORDER BY charttime NULLS FIRST) AS rn
  FROM wt_stg
  WHERE
    NOT weight IS NULL
), wt_stg2 AS (
  SELECT
    wt_stg1.stay_id,
    ie.intime,
    ie.outtime,
    wt_stg1.weight_type,
    CASE
      WHEN wt_stg1.weight_type = 'admit' AND wt_stg1.rn = 1
      THEN ie.intime - INTERVAL '2 HOUR'
      ELSE wt_stg1.charttime
    END AS starttime,
    wt_stg1.weight
  FROM wt_stg1
  INNER JOIN mimiciv_icu.icustays AS ie
    ON ie.stay_id = wt_stg1.stay_id
), wt_stg3 AS (
  SELECT
    stay_id,
    intime,
    outtime,
    starttime,
    COALESCE(
      LEAD(starttime) OVER (PARTITION BY stay_id ORDER BY starttime NULLS FIRST),
      outtime + INTERVAL '2 HOUR'
    ) AS endtime,
    weight,
    weight_type
  FROM wt_stg2
), wt1 AS (
  SELECT
    stay_id,
    starttime,
    COALESCE(
      endtime,
      LEAD(starttime) OVER (PARTITION BY stay_id ORDER BY starttime NULLS FIRST) /* impute ICU discharge as the end of the final weight measurement */ /* plus a 2 hour "fuzziness" window */,
      outtime + INTERVAL '2 HOUR'
    ) AS endtime,
    weight,
    weight_type
  FROM wt_stg3
), wt_fix AS (
  SELECT
    ie.stay_id, /* we add a 2 hour "fuzziness" window */
    ie.intime - INTERVAL '2 HOUR' AS starttime,
    wt.starttime AS endtime,
    wt.weight,
    wt.weight_type
  FROM mimiciv_icu.icustays AS ie
  INNER JOIN (
    SELECT
      wt1.stay_id,
      wt1.starttime,
      wt1.weight,
      weight_type,
      ROW_NUMBER() OVER (PARTITION BY wt1.stay_id ORDER BY wt1.starttime NULLS FIRST) AS rn
    FROM wt1
  ) AS wt
    ON ie.stay_id = wt.stay_id AND wt.rn = 1 AND ie.intime < wt.starttime
)
/* add the backfill rows to the main weight table */
SELECT
  wt1.stay_id,
  wt1.starttime,
  wt1.endtime,
  wt1.weight,
  wt1.weight_type
FROM wt1
UNION ALL
SELECT
  wt_fix.stay_id,
  wt_fix.starttime,
  wt_fix.endtime,
  wt_fix.weight,
  wt_fix.weight_type
FROM wt_fix

-- THIS SCRIPT IS AUTOMATICALLY GENERATED. DO NOT EDIT IT DIRECTLY.
DROP TABLE IF EXISTS mimiciv_derived.urine_output; CREATE TABLE mimiciv_derived.urine_output AS
WITH uo AS (
  SELECT
    oe.stay_id,
    oe.charttime, /* volumes associated with urine output ITEMIDs */ /* note we consider input of GU irrigant as a negative volume */ /* GU irrigant volume in usually has a corresponding volume out */ /* so the net is often 0, despite large irrigant volumes */
    CASE WHEN oe.itemid = 227488 AND oe.value > 0 THEN -1 * oe.value ELSE oe.value END AS urineoutput
  FROM mimiciv_icu.outputevents AS oe
  WHERE
    itemid IN (226559 /* Foley */, 226560 /* Void */, 226561 /* Condom Cath */, 226584 /* Ileoconduit */, 226563 /* Suprapubic */, 226564 /* R Nephrostomy */, 226565 /* L Nephrostomy */, 226567 /* Straight Cath */, 226557 /* R Ureteral Stent */, 226558 /* L Ureteral Stent */, 227488 /* GU Irrigant Volume In */, 227489 /* GU Irrigant/Urine Volume Out */)
)
SELECT
  stay_id,
  charttime,
  SUM(urineoutput) AS urineoutput
FROM uo
GROUP BY
  stay_id,
  charttime

-- THIS SCRIPT IS AUTOMATICALLY GENERATED. DO NOT EDIT IT DIRECTLY.
DROP TABLE IF EXISTS mimiciv_derived.urine_output_rate; CREATE TABLE mimiciv_derived.urine_output_rate AS
/* attempt to calculate urine output per hour */ /* rate/hour is the interpretable measure of kidney function */ /* though it is difficult to estimate from aperiodic point measures */ /* first we get the earliest heart rate documented for the stay */
WITH tm AS (
  SELECT
    ie.stay_id,
    MIN(charttime) AS intime_hr,
    MAX(charttime) AS outtime_hr
  FROM mimiciv_icu.icustays AS ie
  INNER JOIN mimiciv_icu.chartevents AS ce
    ON ie.stay_id = ce.stay_id
    AND ce.itemid = 220045
    AND ce.charttime > ie.intime - INTERVAL '1 MONTH'
    AND ce.charttime < ie.outtime + INTERVAL '1 MONTH'
  GROUP BY
    ie.stay_id
), uo_tm AS (
  SELECT
    tm.stay_id,
    CASE
      WHEN LAG(charttime) OVER w IS NULL
      THEN EXTRACT(EPOCH FROM charttime - intime_hr) / 60.0
      ELSE EXTRACT(EPOCH FROM charttime - LAG(charttime) OVER w) / 60.0
    END AS tm_since_last_uo,
    uo.charttime,
    uo.urineoutput
  FROM tm
  INNER JOIN mimiciv_derived.urine_output AS uo
    ON tm.stay_id = uo.stay_id
  WINDOW w AS (PARTITION BY tm.stay_id ORDER BY charttime NULLS FIRST)
), ur_stg AS (
  SELECT
    io.stay_id,
    io.charttime, /* we have joined each row to all rows preceding within 24 hours */ /* we can now sum these rows to get total UO over the last 24 hours */ /* we can use case statements to restrict it to only the last 6/12 hours */ /* therefore we have three sums: */ /* 1) over a 6 hour period */ /* 2) over a 12 hour period */ /* 3) over a 24 hour period */
    SUM(DISTINCT io.urineoutput) AS uo, /* note that we assume data charted at charttime corresponds */ /* to 1 hour of UO, therefore we use '5' and '11' to restrict the */ /* period, rather than 6/12 this assumption may overestimate UO rate */ /* when documentation is done less than hourly */
    SUM(
      CASE
        WHEN EXTRACT(EPOCH FROM io.charttime - iosum.charttime) / 3600.0 <= 5
        THEN iosum.urineoutput
        ELSE NULL
      END
    ) AS urineoutput_6hr,
    CAST(SUM(
      CASE
        WHEN EXTRACT(EPOCH FROM io.charttime - iosum.charttime) / 3600.0 <= 5
        THEN iosum.tm_since_last_uo
        ELSE NULL
      END
    ) AS DOUBLE PRECISION) / 60.0 AS uo_tm_6hr,
    SUM(
      CASE
        WHEN EXTRACT(EPOCH FROM io.charttime - iosum.charttime) / 3600.0 <= 11
        THEN iosum.urineoutput
        ELSE NULL
      END
    ) AS urineoutput_12hr,
    CAST(SUM(
      CASE
        WHEN EXTRACT(EPOCH FROM io.charttime - iosum.charttime) / 3600.0 <= 11
        THEN iosum.tm_since_last_uo
        ELSE NULL
      END
    ) AS DOUBLE PRECISION) / 60.0 AS uo_tm_12hr, /* 24 hours */
    SUM(iosum.urineoutput) AS urineoutput_24hr,
    CAST(SUM(iosum.tm_since_last_uo) AS DOUBLE PRECISION) / 60.0 AS uo_tm_24hr
  FROM uo_tm AS io
  /* this join gives you all UO measurements over a 24 hour period */
  LEFT JOIN uo_tm AS iosum
    ON io.stay_id = iosum.stay_id
    AND io.charttime >= iosum.charttime
    AND io.charttime <= (
      iosum.charttime + INTERVAL '23 HOUR'
    )
  GROUP BY
    io.stay_id,
    io.charttime
)
SELECT
  ur.stay_id,
  ur.charttime,
  wd.weight,
  ur.uo,
  ur.urineoutput_6hr,
  ur.urineoutput_12hr,
  ur.urineoutput_24hr,
  CASE
    WHEN uo_tm_6hr >= 6
    THEN ROUND(
      CAST((
        CAST(CAST(ur.urineoutput_6hr AS DOUBLE PRECISION) / wd.weight AS DOUBLE PRECISION) / uo_tm_6hr
      ) AS DECIMAL),
      4
    )
  END AS uo_mlkghr_6hr,
  CASE
    WHEN uo_tm_12hr >= 12
    THEN ROUND(
      CAST((
        CAST(CAST(ur.urineoutput_12hr AS DOUBLE PRECISION) / wd.weight AS DOUBLE PRECISION) / uo_tm_12hr
      ) AS DECIMAL),
      4
    )
  END AS uo_mlkghr_12hr,
  CASE
    WHEN uo_tm_24hr >= 24
    THEN ROUND(
      CAST((
        CAST(CAST(ur.urineoutput_24hr AS DOUBLE PRECISION) / wd.weight AS DOUBLE PRECISION) / uo_tm_24hr
      ) AS DECIMAL),
      4
    )
  END AS uo_mlkghr_24hr, /* time of earliest UO measurement that was used to calculate the rate */
  ROUND(CAST(uo_tm_6hr AS DECIMAL), 2) AS uo_tm_6hr,
  ROUND(CAST(uo_tm_12hr AS DECIMAL), 2) AS uo_tm_12hr,
  ROUND(CAST(uo_tm_24hr AS DECIMAL), 2) AS uo_tm_24hr
FROM ur_stg AS ur
LEFT JOIN mimiciv_derived.weight_durations AS wd
  ON ur.stay_id = wd.stay_id
  AND ur.charttime > wd.starttime
  AND ur.charttime <= wd.endtime
  AND wd.weight > 0

--------------------------------------------------------
--------------------------------------------------------

select 
    (select count(*) from mimiciv_derived.weight_durations) as weight_rows,
    (select count(*) from mimiciv_derived.urine_output) as urine_rows,
    (select count(*) from mimiciv_derived.urine_output_rate) as urine_rate_rows;


with cohorte as (
    select c.stay_id, cb.intime
    from public.dataset_final_v3_clean c
    join public.cohorte_base_v3 cb on c.stay_id = cb.stay_id
)
select
    (select count(*) from public.dataset_final_v3_clean) as total_cohorte,
    count(distinct c.stay_id) as con_algun_registro_uo,
    count(distinct case when uo.uo is not null then c.stay_id end) as con_volumen_uo,
    count(distinct case when uo.uo_mlkghr_6hr is not null then c.stay_id end) as con_tasa_mlkgh
from cohorte c
left join mimiciv_derived.urine_output_rate uo
    on c.stay_id = uo.stay_id
    and uo.charttime >= c.intime
    and uo.charttime <= c.intime + interval '6 hour';

--------

drop table if exists public.variables_diuresis_v3;

create table public.variables_diuresis_v3 as
with cohorte as (
    select c.stay_id, cb.intime
    from public.dataset_final_v3_clean c
    join public.cohorte_base_v3 cb on c.stay_id = cb.stay_id
),
-- Volumen total de orina en las primeras 6h, sumando todos los registros
-- de urine_output (ya considera GU irrigant como negativo según script oficial)
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
-- Peso del paciente (el más cercano a intime)
peso_paciente as (
    select distinct on (c.stay_id)
        c.stay_id,
        wd.weight
    from cohorte c
    left join mimiciv_derived.weight_durations wd
        on c.stay_id = wd.stay_id
        and wd.starttime <= c.intime + interval '6 hour'
        and wd.endtime >= c.intime
        and wd.weight > 0
    order by c.stay_id, wd.starttime
)
select
    v.stay_id,
    v.diuresis_volumen_6h,
    v.diuresis_n_registros,
    p.weight as peso_kg,
    -- ml/kg normalizado (sin dividir por tiempo, que no es fiable en 6h)
    case 
        when p.weight > 0 and v.diuresis_volumen_6h is not null 
        then v.diuresis_volumen_6h / p.weight 
        else null 
    end as diuresis_ml_kg_6h
from volumen_6h v
left join peso_paciente p on v.stay_id = p.stay_id;

-- Verificación
select 
    count(*) as total,
    count(diuresis_volumen_6h) as con_volumen,
    count(peso_kg) as con_peso,
    count(diuresis_ml_kg_6h) as con_normalizada,
    round(avg(diuresis_volumen_6h)::numeric, 1) as volumen_medio_ml,
    round(avg(diuresis_ml_kg_6h)::numeric, 2) as ml_kg_medio
from public.variables_diuresis_v3;