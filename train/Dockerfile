FROM python:3.6

# LABEL
LABEL envs="[\"FHIR_SERVER\",\"FHIR_PORT\",\"MODE\"]"

WORKDIR /home

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD [ "python", "./main.py" ]




