import requests
import time
from prometheus_exporter import (
    model_accuracy,
    request_latency,
    prediction_request_total,
    prediction_success_total,
    prediction_failure_total,
    last_prediction_latency,
    last_prediction_score,
    service_uptime,
    concurrent_requests,
    input_average_score,
    start_exporter,
)

start_time = time.time()
start_exporter()

url = "http://127.0.0.1:5000/invocations"

data = {
    "dataframe_records": [
        {
            "math score": 0.390024,
            "reading score": 0.193999,
            "writing score": 0.391492,
            "average_score": 0.343574,
            "gender_male": False,
            "race/ethnicity_group B": True,
            "race/ethnicity_group C": False,
            "race/ethnicity_group D": False,
            "race/ethnicity_group E": False,
            "parental level of education_bachelor's degree": True,
            "parental level of education_high school": False,
            "parental level of education_master's degree": False,
            "parental level of education_some college": False,
            "parental level of education_some high school": False,
            "lunch_standard": True,
            "test preparation course_none": True
        }
    ]
}

model_accuracy.set(0.85)

while True:
    try:
        service_uptime.set(time.time() - start_time)
        avg_score = data["dataframe_records"][0]["average_score"]
        input_average_score.set(avg_score)

        prediction_request_total.inc()
        concurrent_requests.inc()

        start_req = time.time()
        response = requests.post(url, json=data)
        latency = time.time() - start_req

        request_latency.observe(latency)
        last_prediction_latency.set(latency)

        if response.status_code == 200:
            prediction_success_total.inc()
            result = response.json()
            print("Prediksi:", result)

            score_value = 0.0
            if isinstance(result, dict):
                if "predictions" in result and isinstance(result["predictions"], list) and len(result["predictions"]) > 0:
                    try:
                        score_value = float(result["predictions"][0])
                    except Exception:
                        score_value = 0.0
            last_prediction_score.set(score_value)
        else:
            prediction_failure_total.inc()
            print("Gagal prediksi:", response.text)
    except Exception as e:
        prediction_failure_total.inc()
        print("Error:", e)
    finally:
        concurrent_requests.dec()

    time.sleep(5)
