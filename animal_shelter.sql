/* Long Beach Animal Shelter Operations & Adoption Efficiency Analysis 
*/

-- Table Creation
CREATE DATABASE animal_shelter_db;
USE animal_shelter_db;

DROP TABLE IF EXISTS animal_shelter;

CREATE TABLE animal_shelter (
	animal_id VARCHAR(50) PRIMARY KEY,
    animal_type VARCHAR(50),
    sex VARCHAR(50),
    
    intake_date DATE,
    intake_condition VARCHAR(100),
    intake_type VARCHAR(100),
    
    outcome_date DATE,
    
	crossing VARCHAR(100),
    jurisdiction VARCHAR(100),
  
	outcome_type VARCHAR(100),
    intake_duration INT
    );

SELECT * FROM animal_shelter;

-- Data Cleaning
SET SQL_SAFE_UPDATES = 0;
UPDATE animal_shelter
SET intake_date = STR_TO_DATE(intake_date, '%Y-%m-%d')
WHERE intake_date IS NOT NULL;
UPDATE animal_shelter
SET outcome_date = STR_TO_DATE(outcome_date, '%Y-%m-%d')
WHERE outcome_date IS NOT NULL;
UPDATE animal_shelter
SET intake_duration = DATEDIFF(outcome_date, intake_date)
WHERE intake_duration IS NULL;

SELECT *
FROM animal_shelter
WHERE intake_date IS NULL
OR outcome_date IS NULL
OR intake_duration IS NULL;

SELECT MIN(intake_date)
FROM animal_shelter;

SELECT MAX(outcome_date)
FROM animal_shelter;

/* The timeline of the data starting at 2013-07-24 to current (2026-02-24)
*/

-- KPI 1: Total Animals
SELECT COUNT(*) as total_animals
FROM animal_shelter;

/* The total intake animals overtime is 51,224 animals
*/

-- KPI 2: Overall Adoption Rate
SELECT
	COUNT(*) AS total,
    SUM(CASE WHEN outcome_type IN ('ADOPTION', 'FOSTER TO ADOPT', 'FOSTER', 'RETURN TO OWNER', 'RESCUE') THEN 1 ELSE 0 END) AS adopted,
    ROUND(SUM(CASE WHEN outcome_type IN ('ADOPTION', 'FOSTER TO ADOPT', 'FOSTER', 'RETURN TO OWNER', 'RESCUE') THEN 1 ELSE 0 END)*100/COUNT(*), 2) AS adoption_rate
FROM animal_shelter;

/* The total adopted animals is 28,468 animals which equivalent to 55.58%.
*/

-- KPI 3: Adoption Rate by Animal Type and Sex
SELECT
	animal_type,
    COUNT(*) AS total,
    SUM(CASE WHEN outcome_type IN ('ADOPTION', 'FOSTER TO ADOPT', 'FOSTER', 'RETURN TO OWNER', 'RESCUE') THEN 1 ELSE 0 END) AS adopted,
    ROUND(SUM(CASE WHEN outcome_type IN ('ADOPTION', 'FOSTER TO ADOPT', 'FOSTER', 'RETURN TO OWNER', 'RESCUE') THEN 1 ELSE 0 END)*100/COUNT(*), 2) AS adoption_rate
FROM animal_shelter
GROUP BY animal_type
ORDER BY adoption_rate DESC;

SELECT
	animal_type,
    sex,
    COUNT(*) AS total,
    SUM(CASE WHEN outcome_type IN ('ADOPTION', 'FOSTER TO ADOPT', 'FOSTER', 'RETURN TO OWNER', 'RESCUE') THEN 1 ELSE 0 END) AS adopted,
    ROUND(SUM(CASE WHEN outcome_type IN ('ADOPTION', 'FOSTER TO ADOPT', 'FOSTER', 'RETURN TO OWNER', 'RESCUE') THEN 1 ELSE 0 END)*100/COUNT(*), 2) AS adoption_rate
FROM animal_shelter
GROUP BY animal_type, sex
ORDER BY adoption_rate DESC;

/* Adoption rates are highest for livestock (80%), followed by rabbits (77.28%) and dogs (72.59%). Wildlife, which is thought to be returned to wildlife, has the lowest adoption rate.
*/

-- KPI 4: Intake Volume Trend
SELECT
	intake_date,
    COUNT(*) AS intake_volume
FROM animal_shelter
GROUP BY intake_date
ORDER BY intake_date;

SELECT
	intake_date,
    COUNT(*) AS intake_volume
FROM animal_shelter
GROUP BY intake_date
ORDER BY intake_volume DESC;

/* With 74 animals, 2023-05-03 has the highest intake volume.
*/

-- KPI 5: Average Length of Stay by Intake Type
SELECT
	intake_type,
    ROUND(AVG(intake_duration), 2) AS avg_days
FROM animal_shelter
GROUP BY intake_type
ORDER BY avg_days DESC;

/* The longest average stays, approximately 36 days, are for returned animals.
*/

-- KPI 6: Adoption Rate by Intake Condition
SELECT 
	intake_condition,
    COUNT(*) AS total,
    SUM(CASE WHEN outcome_type IN ('ADOPTION', 'FOSTER TO ADOPT', 'FOSTER', 'RETURN TO OWNER', 'RESCUE') THEN 1 ELSE 0 END) AS adopted,
    ROUND(SUM(CASE WHEN outcome_type IN ('ADOPTION', 'FOSTER TO ADOPT', 'FOSTER', 'RETURN TO OWNER', 'RESCUE') THEN 1 ELSE 0 END)*100/COUNT(*), 2) AS adoption_rate
FROM animal_shelter
GROUP BY intake_condition
ORDER BY adoption_rate DESC;

/*
The highest adoption rate is found in welfare seizures, whereas the lowest adoption rate is found in physically ill animals.
*/

-- KPI 7: Animals by Crossings (Geographic Analysis)
SELECT
	crossing,
    COUNT(*) AS animals
FROM animal_shelter
GROUP BY crossing
ORDER BY animals DESC;

/* 
The Crossing at 7700 E Spring St, Long Beach, CA 90815 has the most animals (418).
*/

-- KPI 8: Top Jurisdictions
SELECT
	crossing,
    COUNT(*) AS animals_found
FROM animal_shelter
GROUP BY crossing
ORDER BY animals_found DESC
LIMIT 10;

/*
Long Beach has the most abandoned animals (44202 animals).
*/