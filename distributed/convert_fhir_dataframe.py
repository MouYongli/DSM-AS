from fhirpy import SyncFHIRClient
fhir_server, fhir_port = "137.226.232.119", "8080"
client = SyncFHIRClient('http://{}:{}/fhir'.format(fhir_server, fhir_port))
patients = client.resources('Patient')  # Return lazy search set
i = 0
for patient in patients:
    i = i + 1
    print(i)