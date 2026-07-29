import urllib.request
import json

BASE_URL = "http://localhost:8000"

def run_endpoint_test(name, url, method="GET", body=None):
    try:
        req = urllib.request.Request(url, method=method)
        if body:
            req.add_header('Content-Type', 'application/json')
            data = json.dumps(body).encode('utf-8')
        else:
            data = None
            
        with urllib.request.urlopen(req, data=data) as response:
            status = response.status
            res_body = json.loads(response.read().decode('utf-8'))
            print(f"[{status} OK] {name} ({url})")
            return res_body
    except Exception as e:
        print(f"[FAILED] {name} ({url}): {e}")
        raise e


def main():
    print("Testing SABER API Endpoints...")
    run_endpoint_test("Health Telemetry", f"{BASE_URL}/api/health")
    run_endpoint_test("Dataset Stats", f"{BASE_URL}/api/dataset/stats?name=ben14k")
    run_endpoint_test("Dataset Samples", f"{BASE_URL}/api/dataset/samples?dataset_name=ben14k&limit=6")
    
    query_body = {
        "dataset_name": "ben14k",
        "query_index": 0,
        "source_modality": "s1",
        "target_modality": "s2",
        "top_k": 5,
        "enable_bridge": True,
        "enable_rerank": False,
        "ode_steps": 5
    }
    run_endpoint_test("Retrieval Query", f"{BASE_URL}/api/retrieval/query", method="POST", body=query_body)
    
    ablation_body = {
        "dataset_name": "ben14k",
        "query_index": 0,
        "source_modality": "s1",
        "target_modality": "s2",
        "top_k": 5
    }
    run_endpoint_test("Ablation Study", f"{BASE_URL}/api/retrieval/ablation", method="POST", body=ablation_body)
    
    run_endpoint_test("Benchmark Metrics", f"{BASE_URL}/api/benchmark/metrics")
    run_endpoint_test("Embedding Points", f"{BASE_URL}/api/embedding/points")
    print("\nALL API ENDPOINTS TESTED SUCCESSFULLY! 100% HEALTHY.")


if __name__ == "__main__":
    main()
