from prometheus_client import start_http_server, Gauge, Counter, Histogram

model_accuracy = Gauge("model_accuracy", "Model accuracy")
request_latency = Histogram("request_latency_seconds", "Request latency in seconds")
prediction_request_total = Counter("prediction_request_total", "Total prediction requests")
prediction_success_total = Counter("prediction_success_total", "Total successful predictions")
prediction_failure_total = Counter("prediction_failure_total", "Total failed predictions")
last_prediction_latency = Gauge("last_prediction_latency_seconds", "Latency of last prediction in seconds")
last_prediction_score = Gauge("last_prediction_score", "Score of last prediction")
service_uptime = Gauge("service_uptime_seconds", "Service uptime in seconds")
concurrent_requests = Gauge("concurrent_requests", "Number of concurrent requests")
input_average_score = Gauge("input_average_score", "Average score of last request")

def start_exporter():
    start_http_server(8000)

if __name__ == "__main__":
    import time
    start_exporter()
    start_time = time.time()
    while True:
        service_uptime.set(time.time() - start_time)
        time.sleep(1)
