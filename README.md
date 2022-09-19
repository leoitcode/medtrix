![](images/medtrix.png)

This is a project to generate new medical reports from free text, including AI-Generated sections and patient reports.

<br>

## Application
This application is live with Streamlit to make your own experiment:  
[MedtriX Streamlit](http://ec2-174-129-135-229.compute-1.amazonaws.com:6565/)


## Dataset
In order to install, you need the MIMIC-III dataset (NOTEEVENTS.csv) Patient's Discharges, to get 3 datasets:
- df_struct.csv (MIMIC-III Discharges)
- df_struct_lemma.csv (MIMIC-III lemmatization)
- df_struct_fam.csv (MIMIC-III Family History)

### MIMIC-III

MIMIC-III is a large, freely-available database comprising deidentified health-related data associated with over forty thousand patients who stayed in critical care units of the Beth Israel Deaconess Medical Center between 2001 and 2012. The database includes information such as demographics, vital sign measurements made at the bedside (~1 data point per hour), laboratory test results, procedures, medications, caregiver notes, imaging reports, and mortality (including post-hospital discharge).

You can read more about MIMIC-IV from the following resources:

* [The MIMIC-III PhysioNet project page](https://physionet.org/content/mimiciii/1.4/)
* [The MIMIC-III online documentation](https://mimic.mit.edu/)

## Execution

To execute the application run:
```
docker build -t medtrix .
docker run medtrix
```

## Versioning

Experiments: Conda Environment (environment.yml)

```
notebook/
```
  
Script: Poetry (pyproject.toml)
```
src/
```