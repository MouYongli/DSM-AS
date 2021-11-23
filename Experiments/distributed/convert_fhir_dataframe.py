import pandas as pd
from fhirpy import SyncFHIRClient

X_FEATURES = ['radius_mean', 'texture_mean', 'perimeter_mean',
       'area_mean', 'smoothness_mean', 'compactness_mean', 'concavity_mean',
       'concave.points_mean', 'symmetry_mean', 'fractal_dimension_mean',
       'radius_se', 'texture_se', 'perimeter_se', 'area_se', 'smoothness_se',
       'compactness_se', 'concavity_se', 'concave.points_se', 'symmetry_se',
       'fractal_dimension_se', 'radius_worst', 'texture_worst',
       'perimeter_worst', 'area_worst', 'smoothness_worst',
       'compactness_worst', 'concavity_worst', 'concave.points_worst',
       'symmetry_worst', 'fractal_dimension_worst']

Y_FEATURE = 'label'

fhir_server, fhir_port = "137.226.232.119", "8080"
client = SyncFHIRClient('http://{}:{}/fhir'.format(fhir_server, fhir_port))
patients = client.resources('Patient')  # Return lazy search set
patients_data = []
for patient in patients:
    patient_birthDate = None
    try:
        patient_birthDate = patient.birthDate
    except:
        pass
    patients_data.append([patient.id, patient.gender, patient_birthDate])
patients_df = pd.DataFrame(patients_data, columns=["patient_id", "gender", "birthDate"])
patients_observation = {}
observations = client.resources("Observation").include("Patient", "subject")
for observation in observations:
    try:
        feature = observation["category"][0]["coding"][0]["code"]
        if feature in X_FEATURES:
            value = observation["valueQuantity"]["value"]
            patient_id_str = observation["subject"]["reference"]
            if patient_id_str[:7] == "Patient":
                patient_id = patient_id_str[8:]
                if patient_id not in patients_observation:
                    patients_observation[patient_id] = {}
                patients_observation[patient_id][feature] = float(value)
    except KeyError:
        print("Key error encountered, skipping Observation...")
for k in patients_observation.keys():
    patients_observation[k].update(patient_id=k)
observation_df = pd.DataFrame.from_dict(patients_observation.values())
observation_df.set_index(["patient_id"])

patients_condition = []
conditions = client.resources("Condition")
for condition in conditions:
    try:
        label = condition["code"]["coding"][0]["code"]
        patient_id_str = condition["subject"]["reference"]
        if patient_id_str[:7] == "Patient":
            patient_id = patient_id_str[8:]
            patients_condition.append([patient_id, label])
    except KeyError:
        print("Key error encountered, skipping Condition...")
condition_df = pd.DataFrame(patients_condition, columns=["patient_id", "label"])
final_df = pd.merge(pd.merge(patients_df, observation_df, on="patient_id", how="outer"), condition_df, on="patient_id", how="outer")
final_df.to_csv("./results/data.csv", index=False)
# final_df = pd.read_csv("./results/data.csv")
# final_df = pd.merge(pd.read_csv("./results/uka.csv"), final_df, on="patient_id", how="inner")
# print(len(final_df))

